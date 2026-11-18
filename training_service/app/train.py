import json
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
     root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42
NUMERIC = ["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt", "OverallQual"]
CATEGORICAL = ["MSZoning"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "SalePrice"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ames_train.csv"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house-price")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(root_mean_squared_error(y_test, preds)),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
    }


def build_ridge_baseline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", Ridge(alpha=1.0, random_state=SEED)),
    ])


def build_model(params: dict) -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", TransformedTargetRegressor(
            regressor=XGBRegressor(**params, random_state=SEED, n_jobs=2),
            func=np.log1p,
            inverse_func=np.expm1,
        )),
    ])


def compute_data_stats(df: pd.DataFrame) -> tuple[dict, plt.Figure]:
    y = df[TARGET]
    y_log = np.log1p(y)
    skew_before = float(y.skew())
    skew_after = float(y_log.skew())

    stats = {
        "row_count": int(len(df)),
        "target_skewness_raw": round(skew_before, 4),
        "target_skewness_log1p": round(skew_after, 4),
        "missing_fractions": {
            col: float(df[col].isna().mean()) for col in FEATURES
        },
        "numeric_summary": {
            col: {
                "mean": round(float(df[col].mean()), 2),
                "std": round(float(df[col].std()), 2),
                "min": float(df[col].min()),
                "median": float(df[col].median()),
                "max": float(df[col].max()),
            }
            for col in NUMERIC
        },
        "categorical_frequencies": {
            col: df[col].value_counts(dropna=False).to_dict()
            for col in CATEGORICAL
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(y, bins=35, color="#3b82f6", edgecolor="black", alpha=0.8)
    axes[0].set_title(f"SalePrice Raw (skew={skew_before:.2f})")
    axes[0].set_xlabel("SalePrice (USD)")
    axes[1].hist(y_log, bins=35, color="#10b981", edgecolor="black", alpha=0.8)
    axes[1].set_title(f"log1p(SalePrice) (skew={skew_after:.2f})")
    axes[1].set_xlabel("log1p(SalePrice)")
    fig.tight_layout()
    return stats, fig


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    raw_df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])
    df = raw_df[FEATURES + [TARGET]].copy()

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    dataset = mlflow.data.from_pandas(
        df, source=str(DATA_PATH), targets=TARGET, name="ames_housing"
    )
    stats, fig = compute_data_stats(df)

    # 1. R2.1: Baseline Ridge Run
    with mlflow.start_run(run_name="ridge-baseline") as baseline_run:
        mlflow.log_input(dataset, context="train")
        mlflow.log_params({"model_type": "Ridge", "alpha": 1.0, "seed": SEED})
        ridge_model = build_ridge_baseline()
        ridge_model.fit(X_train, y_train)
        base_metrics = evaluate(ridge_model, X_test, y_test)
        mlflow.log_metrics(base_metrics)
        mlflow.log_dict(stats, "data_stats.json")
        sig_base = infer_signature(X_train, ridge_model.predict(X_train))
        mlflow.sklearn.log_model(ridge_model, name="model", signature=sig_base, serialization_format="cloudpickle")
        v_base = mlflow.register_model(
            f"runs:/{baseline_run.info.run_id}/model", MODEL_NAME
        )
        print(f"[trainer] Baseline Ridge registered as v{v_base.version}: {base_metrics}")

    # 2. R2.4 & R2.5: Optuna Hyperparameter Search (Parent + 20 Nested Child Runs)
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            pipe = build_model(params)
            scores = cross_val_score(
                pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=1
            )
            cv_r2 = float(scores.mean())
            mlflow.log_metric("cv_r2", cv_r2)
            return cv_r2

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    with mlflow.start_run(run_name="xgboost-optuna-search"):
        mlflow.log_param("n_trials", 20)
        mlflow.log_param("sampler", "TPESampler")
        mlflow.log_param("seed", SEED)
        study.optimize(objective, n_trials=20)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_r2", float(study.best_value))
        print(f"[trainer] Optuna best CV R2={study.best_value:.4f}, params={study.best_params}")

    # 3. R2.6–R2.11: Final XGBoost Run + Artifacts + Model Registry + @prod Alias
    with mlflow.start_run(run_name="xgboost-final") as run:
        mlflow.log_input(dataset, context="train")
        final_model = build_model(study.best_params)
        final_model.fit(X_train, y_train)

        mlflow.log_params(study.best_params)
        final_metrics = evaluate(final_model, X_test, y_test)
        mlflow.log_metrics(final_metrics)
        mlflow.log_metrics({
            "row_count": stats["row_count"],
            "target_skewness_raw": stats["target_skewness_raw"],
            "target_skewness_log1p": stats["target_skewness_log1p"],
        })

        mlflow.log_dict(stats, "data_stats.json")
        mlflow.log_figure(fig, "target_distributions.png")
        plt.close(fig)

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference_data.parquet"
            X_train.to_parquet(ref_path, index=False)
            mlflow.log_artifact(str(ref_path))

        signature = infer_signature(X_train, final_model.predict(X_train))
        mlflow.sklearn.log_model(final_model, name="model", signature=signature, serialization_format="cloudpickle")

        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", MODEL_NAME
        )
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)
        print(
            f"[trainer] Final XGBoost registered as v{version.version} (@{MODEL_ALIAS}): {final_metrics}"
        )


if __name__ == "__main__":
    main()