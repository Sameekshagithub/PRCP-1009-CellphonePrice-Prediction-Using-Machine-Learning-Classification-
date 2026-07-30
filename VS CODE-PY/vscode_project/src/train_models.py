"""
train_models.py
----------------
Trains and evaluates six classification algorithms for predicting
`price_range`, each in its own clearly separated function:

    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - K-Nearest Neighbours (KNN)
    - Support Vector Machine (SVM)
    - Gradient Boosting Classifier

All comparison charts and confusion matrices are saved under outputs/.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CLASS_NAMES = ["Low(0)", "Medium(1)", "High(2)", "Very High(3)"]

# Models that need scaled features
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM"}

OUTPUT_DIR = "outputs"
CM_DIR = f"{OUTPUT_DIR}/confusion_matrices"
MODELS_DIR = "models"


def prepare_data(df: pd.DataFrame):
    """Splits features/target, then train/test splits and scales."""
    X = df.drop(columns=["price_range"])
    y = df["price_range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def _evaluate_and_log(model_name, model, y_test, preds, results, trained_models):
    """Shared evaluation + confusion-matrix-saving logic for every model."""
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="weighted"),
        "Recall": recall_score(y_test, preds, average="weighted"),
        "F1-Score": f1_score(y_test, preds, average="weighted"),
    }
    trained_models[model_name] = model

    print(f"\n--- {model_name} ---")
    print(pd.Series(results[model_name]).round(4))
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    os.makedirs(CM_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(f"{CM_DIR}/{safe_name}.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# One function per classifier — each trains, evaluates, and logs its model.
# --------------------------------------------------------------------------

def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test,
                                results, trained_models):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    _evaluate_and_log("Logistic Regression", model, y_test, preds,
                       results, trained_models)


def train_decision_tree(X_train, X_test, y_train, y_test,
                         results, trained_models):
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _evaluate_and_log("Decision Tree", model, y_test, preds,
                       results, trained_models)


def train_random_forest(X_train, X_test, y_train, y_test,
                         results, trained_models):
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _evaluate_and_log("Random Forest", model, y_test, preds,
                       results, trained_models)


def train_knn(X_train_scaled, X_test_scaled, y_train, y_test,
              results, trained_models):
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    _evaluate_and_log("KNN", model, y_test, preds, results, trained_models)


def train_svm(X_train_scaled, X_test_scaled, y_train, y_test,
              results, trained_models):
    model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    _evaluate_and_log("SVM", model, y_test, preds, results, trained_models)


def train_gradient_boosting(X_train, X_test, y_train, y_test,
                             results, trained_models):
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _evaluate_and_log("Gradient Boosting", model, y_test, preds,
                       results, trained_models)


def train_all_models(X_train, X_test, y_train, y_test,
                      X_train_scaled, X_test_scaled):
    """Calls every classifier's training function in turn."""
    results, trained_models = {}, {}

    train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test,
                               results, trained_models)
    train_decision_tree(X_train, X_test, y_train, y_test,
                         results, trained_models)
    train_random_forest(X_train, X_test, y_train, y_test,
                         results, trained_models)
    train_knn(X_train_scaled, X_test_scaled, y_train, y_test,
              results, trained_models)
    train_svm(X_train_scaled, X_test_scaled, y_train, y_test,
              results, trained_models)
    train_gradient_boosting(X_train, X_test, y_train, y_test,
                             results, trained_models)

    return results, trained_models


def compare_models(results: dict) -> pd.DataFrame:
    """Builds the comparison table + bar chart, saves both to outputs/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")

    print("\n===== MODEL COMPARISON =====")
    print(results_df.round(4))

    fig, ax = plt.subplots(figsize=(10, 6))
    results_df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
    plt.close(fig)

    return results_df


def save_best_model(results_df, trained_models, scaler, feature_columns):
    """Saves the best model (by accuracy) as a bundle usable by the Flask app."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_name = results_df["Accuracy"].idxmax()

    bundle = {
        "model": trained_models[best_model_name],
        "scaler": scaler,
        "feature_columns": list(feature_columns),
        "needs_scaling": best_model_name in SCALED_MODELS,
        "model_name": best_model_name,
    }
    with open(f"{MODELS_DIR}/model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[train_models] Best model: {best_model_name} "
          f"(Accuracy={results_df.loc[best_model_name, 'Accuracy']:.4f})")
    print(f"[train_models] Saved model bundle to '{MODELS_DIR}/model_bundle.pkl'")
    return best_model_name


if __name__ == "__main__":
    try:
        from src.data_loader import load_data, basic_checks
    except ImportError:
        from data_loader import load_data, basic_checks

    df = load_data()
    basic_checks(df)

    (X, y, X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, scaler) = prepare_data(df)

    results, trained_models = train_all_models(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )
    results_df = compare_models(results)
    save_best_model(results_df, trained_models, scaler, X.columns)
