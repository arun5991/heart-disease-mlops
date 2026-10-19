import os
import sys
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processor import (  # noqa: E402
    FEATURE_COLUMNS,
    build_preprocessor,
    load_and_clean_data,
)


def plot_and_save_eval(y_true, y_prob, y_pred, model_name, output_dir="artifacts/plots"):
    os.makedirs(output_dir, exist_ok=True)
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(output_dir, f"cm_{model_name}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, f"roc_{model_name}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return cm_path, roc_path


def train_and_evaluate():
    df = load_and_clean_data()
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Heart_Disease_Classification_MLOps")

    models_config = {
        "Logistic_Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 6, 10],
                "classifier__min_samples_split": [2, 5],
            },
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_auc = -1.0
    best_overall_pipeline = None
    best_model_name = ""
    summary_results = {}

    for model_name, cfg in models_config.items():
        with mlflow.start_run(run_name=model_name):
            full_pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", cfg["estimator"]),
            ])

            grid_search = GridSearchCV(
                full_pipeline,
                param_grid=cfg["param_grid"],
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            best_pipe = grid_search.best_estimator_
            y_pred = best_pipe.predict(X_test)
            y_prob = best_pipe.predict_proba(X_test)[:, 1]

            metrics = {
                "cv_roc_auc_mean": float(grid_search.best_score_),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
            }

            # Log parameters, metrics, plots, and model to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)

            cm_path, roc_path = plot_and_save_eval(y_test, y_prob, y_pred, model_name)
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
            mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")

            mlflow.sklearn.log_model(
                sk_model=best_pipe,
                artifact_path="model",
                input_example=X_train.iloc[:2],
            )

            summary_results[model_name] = {
                "best_params": grid_search.best_params_,
                "metrics": metrics,
            }

            print(f"\n--- {model_name} Results ---")
            print(f"Best Params: {grid_search.best_params_}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            if metrics["test_roc_auc"] > best_overall_auc:
                best_overall_auc = metrics["test_roc_auc"]
                best_overall_pipeline = best_pipe
                best_model_name = model_name

    # Package winning model in reusable format (Task 4)
    os.makedirs("artifacts/models", exist_ok=True)
    model_path = "artifacts/models/heart_disease_pipeline.joblib"
    joblib.dump(best_overall_pipeline, model_path)

    with open("artifacts/models/metrics_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print(
        f"\nWinner: {best_model_name} (Test ROC-AUC: {best_overall_auc:.4f}) "
        f"saved to {model_path}"
    )


if __name__ == "__main__":
    train_and_evaluate()