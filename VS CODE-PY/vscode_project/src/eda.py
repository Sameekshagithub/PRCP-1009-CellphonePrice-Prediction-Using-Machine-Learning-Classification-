"""
eda.py
------
Exploratory Data Analysis for the cellphone price dataset. Every plot is
saved to disk under outputs/eda/ (no plt.show(), so this runs headlessly
in a terminal / CI / VS Code "Run Python File" without blocking).
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless backend - never blocks on a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

OUTPUT_DIR = "outputs/eda"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def class_balance(df: pd.DataFrame):
    _ensure_output_dir()
    counts = df["price_range"].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    print("\n===== CLASS BALANCE (price_range) =====")
    print(pd.DataFrame({"count": counts, "percentage": pct}))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="price_range", data=df, palette="viridis", ax=ax)
    ax.set_title("Distribution of Price Range Classes")
    ax.set_xlabel("Price Range (0=Low, 1=Medium, 2=High, 3=Very High)")
    ax.set_ylabel("Number of Phones")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/class_balance.png", dpi=150)
    plt.close(fig)


def summary_statistics(df: pd.DataFrame):
    _ensure_output_dir()
    summary = df.describe().T
    summary["range"] = summary["max"] - summary["min"]
    summary = summary.sort_values("range", ascending=False)
    summary.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv")
    print("\n===== SUMMARY STATISTICS (saved to summary_statistics.csv) =====")
    print(summary.head(10))


def feature_distributions(df: pd.DataFrame):
    _ensure_output_dir()
    key_features = ["battery_power", "ram", "int_memory", "px_height",
                     "px_width", "mobile_wt"]
    key_features = [f for f in key_features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), key_features):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/feature_distributions.png", dpi=150)
    plt.close(fig)


def outlier_detection(df: pd.DataFrame) -> pd.Series:
    _ensure_output_dir()
    continuous_cols = ["battery_power", "clock_speed", "fc", "int_memory",
                        "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
                        "px_width", "ram", "sc_h", "sc_w", "talk_time"]
    continuous_cols = [c for c in continuous_cols if c in df.columns]

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    for ax, col in zip(axes.flatten(), continuous_cols):
        sns.boxplot(y=df[col], ax=ax, color="lightcoral")
        ax.set_title(col)
    for ax in axes.flatten()[len(continuous_cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/outlier_boxplots.png", dpi=150)
    plt.close(fig)

    outlier_report = {}
    for col in continuous_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_report[col] = int(((df[col] < lower) | (df[col] > upper)).sum())

    outlier_series = pd.Series(outlier_report, name="outlier_count").sort_values(
        ascending=False
    )
    print("\n===== OUTLIER COUNTS (IQR rule) =====")
    print(outlier_series)
    return outlier_series


def correlation_heatmap(df: pd.DataFrame):
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3, ax=ax)
    ax.set_title("Correlation Heatmap — All Features")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print("\n===== TOP CORRELATIONS WITH price_range =====")
    print(corr["price_range"].sort_values(ascending=False))


def run_full_eda(df: pd.DataFrame):
    """Runs every EDA step and saves all charts under outputs/eda/."""
    class_balance(df)
    summary_statistics(df)
    feature_distributions(df)
    outlier_detection(df)
    correlation_heatmap(df)
    print(f"\n[eda] All EDA charts saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    try:
        from src.data_loader import load_data, basic_checks
    except ImportError:
        from data_loader import load_data, basic_checks

    df = load_data()
    basic_checks(df)
    run_full_eda(df)
