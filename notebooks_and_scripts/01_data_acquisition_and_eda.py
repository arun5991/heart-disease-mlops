import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processor import load_and_clean_data, NUMERIC_FEATURES  # noqa: E402


def run_eda(output_dir: str = "artifacts/plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean_data()

    print("=== Dataset Shape & Info ===")
    print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    print("\nMissing values after cleaning:\n", df.isnull().sum())
    print("\nTarget Class Distribution:\n", df["target"].value_counts())

    sns.set_theme(style="whitegrid")

    # 1. Class Balance Bar Plot
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="target", hue="target", palette="viridis", legend=False)
    ax.set_title("Heart Disease Class Balance (0 = Healthy, 1 = Disease)", fontsize=12)
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Patient Count")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_balance.png"), dpi=150)
    plt.close()

    # 2. Continuous Features Histograms & KDE
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, col in enumerate(NUMERIC_FEATURES):
        sns.histplot(
            data=df, x=col, hue="target", kde=True, palette="crest", ax=axes[idx], element="step"
        )
        axes[idx].set_title(f"Distribution of {col} by Target")
    axes[-1].axis("off")
    plt.suptitle("Continuous Health Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_feature_histograms.png"), dpi=150)
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
    plt.title("Feature Correlation Matrix - UCI Heart Disease Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_heatmap.png"), dpi=150)
    plt.close()

    print(f"\nEDA plots saved successfully to '{output_dir}/'")


if __name__ == "__main__":
    run_eda()