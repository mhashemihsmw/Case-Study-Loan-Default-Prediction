"""Modeling utilities for the loan default project."""

from __future__ import annotations

from typing import Iterable, Sequence
import re
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import re


def train_test_splits(
    frame: pd.DataFrame, rn_state: int, test_size: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test partitions while keeping a copy intact."""

    data = frame.copy()
    y = data.pop("Status")
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=test_size, random_state=rn_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def mlflow_metrics(
    y_train: pd.Series,
    y_train_pred: Iterable,
    y_test: pd.Series,
    y_test_pred: Iterable,
) -> None:
    """Log core classification metrics to mlflow."""

    mlflow.log_metric("Train Accuracy", accuracy_score(y_train, y_train_pred))
    mlflow.log_metric("Train Precision", precision_score(y_train, y_train_pred))
    mlflow.log_metric("Train Recall", recall_score(y_train, y_train_pred))
    mlflow.log_metric("Train ROC/AUC", roc_auc_score(y_train, y_train_pred))

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_test_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_test_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_test_pred))
    mlflow.log_metric("Test ROC/AUC", roc_auc_score(y_test, y_test_pred))


def build_preprocessor(numeric_features: Sequence[str]) -> ColumnTransformer:
    """Create preprocessing pipeline with MinMax scaling on numeric features."""

    numeric_transformer = Pipeline(steps=[("Scaler", MinMaxScaler())])
    return ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])


def next_experiment_name(prefix: str = "loan-default-exp", width: int = 3) -> str:
    """Return the next experiment name by incrementing the numeric suffix if present."""

    experiments = mlflow.search_experiments()
    pattern = re.compile(rf"{re.escape(prefix)}[-_]?(\d+)$")
    max_num = 0

    for exp in experiments:
        if not exp.name.startswith(prefix):
            continue
        match = pattern.search(exp.name)
        if match:
            max_num = max(max_num, int(match.group(1)))
        elif exp.name == prefix:
            max_num = max(max_num, 0)

    return f"{prefix}-{max_num + 1:0{width}d}"


def mlflow_models(
    flows: list[list],
    train_with_dummies: pd.DataFrame,
    test_with_dummies: pd.DataFrame,
    train_without_dummies: pd.DataFrame,
    test_without_dummies: pd.DataFrame,
    numeric_features: Sequence[str],
    cv: int = 7,
    scoring: list[str] | None = None,
    verbose: int = 3,
    experiment_name: str = "loan-default-exp",
    tracking_uri: str | None = None,
    experiment_name_autoincrement: bool = False,
    experiment_name_width: int = 3,
) -> None:
    """Grid-search multiple models and log results with mlflow.

    Expects pre-split datasets so that preprocessing/imputation can be fit on
    the training portion only, avoiding target leakage.
    """

    if scoring is None:
        scoring = ["accuracy", "precision", "recall", "roc_auc"]

    names, models, use_dummies_flags, param_grids = flows

    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name_autoincrement:
        experiment_name = next_experiment_name(prefix=experiment_name, width=experiment_name_width)

    for run_idx, (name, model, use_dummies, params) in enumerate(
        zip(names, models, use_dummies_flags, param_grids), start=1
    ):
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("Model", name)
            mlflow.set_tag("RunIndex", run_idx)
            mlflow.log_param("PreSplit", True)
            mlflow.log_param("UseDummies", bool(use_dummies))
            mlflow.log_param("CVFolds", cv)
            mlflow.log_param("Scoring", ",".join(scoring))
            mlflow.set_experiment_tag("ExpID", f"{experiment_name}.{name}.run_{run_idx}")
            mlflow.set_tag("NumericFeatures", ",".join(numeric_features))

            dataset_train = train_with_dummies.copy() if use_dummies else train_without_dummies.copy()
            dataset_test = test_with_dummies.copy() if use_dummies else test_without_dummies.copy()
            params_prefixed = {f"{name}__{key}": val for key, val in params.items()}

            X_train = dataset_train.drop(columns=["Status"])
            y_train = dataset_train["Status"]
            X_test = dataset_test.drop(columns=["Status"])
            y_test = dataset_test["Status"]

            preprocessor = build_preprocessor(numeric_features)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), (name, model)])

            grid = GridSearchCV(
                pipeline,
                params_prefixed,
                scoring=scoring,
                refit="roc_auc",
                return_train_score=True,
                cv=cv,
                verbose=verbose,
            )

            grid.fit(X_train, y_train)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("CVBestScore", grid.best_score_)
            mlflow.log_param("RefitMetric", grid.refit)
            best_model = grid.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            mlflow_metrics(y_train, y_train_pred, y_test, y_test_pred)

            # Log full CV results as an artifact for inspection.
            cv_df = pd.DataFrame(grid.cv_results_)
            with tempfile.NamedTemporaryFile(suffix="_cv_results.csv", delete=False) as tmp:
                cv_df.to_csv(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, artifact_path="cv_results")

            signature = infer_signature(model_input=X_train, model_output=y_test_pred)
            mlflow.sklearn.log_model(best_model, name="model", signature=signature)
