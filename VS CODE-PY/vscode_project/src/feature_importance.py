"""
feature_importance.py
----------------------
Extracts and plots feature importance from the Random Forest model to
answer: "Which mobile specifications influence the price the most?"
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = "outputs"


def plot_feature_importance(rf_model, feature_names) -> pd.Series:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", header=["importance"])

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(x=importances.values, y=importances.index, palette="mako", ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)

    print("\n===== FEATURE IMPORTANCE =====")
    print(importances)

    print("\nHighly important features:")
    print(importances.head(4))
    print("\nLeast important features:")
    print(importances.tail(4))

    return importances


if __name__ == "__main__":
    # Standalone usage: loads the model bundle saved by train_models.py
    import pickle

    with open("models/model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"The saved best model ('{bundle['model_name']}') doesn't expose "
            "feature_importances_. Run train_models.py first, or inspect the "
            "Random Forest model specifically."
        )

    plot_feature_importance(model, bundle["feature_columns"])
