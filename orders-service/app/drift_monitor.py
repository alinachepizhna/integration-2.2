from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from scipy.stats import ks_2samp

from .database import SessionLocal
from .models import Order

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MONITORING_EXPERIMENT = os.getenv("MLFLOW_MONITORING_EXPERIMENT", "house-price-monitoring")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.5"))
NUMERIC = ["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt", "OverallQual"]


def run_drift_monitoring_job() -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MONITORING_EXPERIMENT)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    ref_path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="reference_data.parquet")
    reference_df = pd.read_parquet(ref_path)

    db = SessionLocal()
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        orders = db.query(Order).filter(Order.created_at >= since).all()
        current_df = pd.DataFrame([
            {
                "GrLivArea": o.gr_liv_area,
                "TotalBsmtSF": o.total_bsmt_sf,
                "GarageArea": o.garage_area,
                "YearBuilt": o.year_built,
                "OverallQual": o.overall_qual,
                "MSZoning": o.ms_zoning,
            }
            for o in orders
        ])
    finally:
        db.close()

    if current_df.empty:
        return {"status": "skipped_no_recent_orders"}

    drifted_cols = []
    details = {}
    for col in NUMERIC:
        stat, p_val = ks_2samp(reference_df[col], current_df[col])
        is_drifted = bool(p_val < 0.05)
        if is_drifted:
            drifted_cols.append(col)
        details[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p_val), 4), "drifted": is_drifted}

    drift_share = round(len(drifted_cols) / len(reference_df.columns), 4)
    report_data = {
        "model_version": mv.version,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "share_of_drifted_features": drift_share,
        "drifted_columns": drifted_cols,
        "column_drift_details": details,
    }

    with mlflow.start_run(run_name=f"drift-check-v{mv.version}") as run:
        mlflow.log_params({"model_version": mv.version, "current_rows": len(current_df)})
        mlflow.log_metric("share_of_drifted_features", drift_share)
        exceeded = drift_share > DRIFT_THRESHOLD
        mlflow.set_tag("drift_exceeded", str(exceeded).lower())

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "drift_report.json"
            html_path = Path(tmpdir) / "drift_report.html"
            json_path.write_text(json.dumps(report_data, indent=2))
            rows_html = "".join(
                f"<tr><td>{c}</td><td>{d['ks_stat']}</td><td>{d['p_value']}</td><td>{d['drifted']}</td></tr>"
                for c, d in details.items()
            )
            html_path.write_text(
                f"<html><body><h2>Data Drift Report (Model v{mv.version})</h2>"
                f"<p>Share of drifted features: <b>{drift_share}</b> (threshold: {DRIFT_THRESHOLD})</p>"
                f"<table border='1' cellpadding='6'><tr><th>Feature</th><th>KS Stat</th><th>p-value</th><th>Drifted</th></tr>"
                f"{rows_html}</table></body></html>"
            )
            mlflow.log_artifact(str(json_path))
            mlflow.log_artifact(str(html_path))

        return {"run_id": run.info.run_id, "share_of_drifted_features": drift_share}


if __name__ == "__main__":
    print(run_drift_monitoring_job())
