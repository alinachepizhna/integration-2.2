from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from .database import SessionLocal
from .models import Order

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MONITORING_EXPERIMENT = os.getenv("MLFLOW_MONITORING_EXPERIMENT", "house-price-monitoring")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.5"))


def run_drift_monitoring_job() -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MONITORING_EXPERIMENT)
    client = MlflowClient()

    # 1. Завантажуємо reference_data.parquet поточної @prod моделі
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    ref_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="reference_data.parquet"
    )
    reference_df = pd.read_parquet(ref_path)

    # 2. Вибираємо замовлення за останні 24 години з PostgreSQL
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

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report_dict = report.as_dict()
    drift_share = float(
        report_dict["metrics"][0]["result"]["share_of_drifted_columns"]
    )

    with mlflow.start_run(run_name=f"drift-check-v{mv.version}") as run:
        mlflow.log_param("model_version", mv.version)
        mlflow.log_param("current_rows", len(current_df))
        mlflow.log_metric("share_of_drifted_features", drift_share)

        exceeded = drift_share > DRIFT_THRESHOLD
        mlflow.set_tag("drift_alert", str(exceeded).lower())
        if exceeded:
            print(f"[WARNING] Data drift threshold exceeded: {drift_share:.2f} > {DRIFT_THRESHOLD}")

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "drift_report.html"
            json_path = Path(tmpdir) / "drift_report.json"
            report.save_html(str(html_path))
            report.save_json(str(json_path))
            mlflow.log_artifact(str(html_path))
            mlflow.log_artifact(str(json_path))

        return {"run_id": run.info.run_id, "share_of_drifted_features": drift_share}


if __name__ == "__main__":
    print(run_drift_monitoring_job())