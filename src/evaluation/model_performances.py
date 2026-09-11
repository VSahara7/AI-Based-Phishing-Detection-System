import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "results" / "extended"

models = [
    "Logistic Regression",
    "Naive Bayes",
    "SVM",
    "Random Forest"
]

accuracy = [0.9743, 0.9448, 0.9781, 0.9806]
precision = [0.9766, 0.9665, 0.9807, 0.9857]
recall = [0.9724, 0.9226, 0.9759, 0.9758]
specificity = [0.9762, 0.9674, 0.9804, 0.9855]
f1 = [0.9745, 0.9441, 0.9783, 0.9807]
balanced_accuracy = [0.9743, 0.9450, 0.9781, 0.9807]

x = np.arange(len(models))
width = 0.13

fig, ax = plt.subplots(figsize=(14, 8))

# Create bars
bars1 = ax.bar(x - 2.5*width, accuracy, width, label="Accuracy")
bars2 = ax.bar(x - 1.5*width, precision, width, label="Precision")
bars3 = ax.bar(x - 0.5*width, recall, width, label="Recall")
bars4 = ax.bar(x + 0.5*width, specificity, width, label="Specificity")
bars5 = ax.bar(x + 1.5*width, f1, width, label="F1-Score")
bars6 = ax.bar(x + 2.5*width, balanced_accuracy, width, label="Balanced Accuracy")

all_bars = [bars1, bars2, bars3, bars4, bars5, bars6]

# ------------------------------------------------
# Highlight Random Forest
# ------------------------------------------------

# Make Random Forest bars visually stronger
for bars in all_bars:
    for i, bar in enumerate(bars):
        if i == 3:  # Random Forest
            bar.set_edgecolor("black")
            bar.set_linewidth(2.5)
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.65)

# Add a background highlight behind Random Forest
rf_x = x[3]

ax.axvspan(
    rf_x - 0.47,
    rf_x + 0.47,
    alpha=0.10,
    zorder=0
)

# ------------------------------------------------
# Labels and formatting
# ------------------------------------------------

ax.set_xlabel(
    "Machine Learning Model",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Performance Score",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Comparison of Machine Learning Models",
    fontsize=15,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)

ax.set_ylim(0.90, 1.00)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

# Legend
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3
)

# ------------------------------------------------
# Add values above bars
# ------------------------------------------------

for bars in all_bars:
    for i, bar in enumerate(bars):

        height = bar.get_height()

        ax.annotate(
            f"{height:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold" if i == 3 else "normal",
            rotation=90
        )

# ------------------------------------------------
# Random Forest annotation
# ------------------------------------------------

ax.annotate(
    "Selected model",
    xy=(rf_x, 0.985),
    xytext=(rf_x, 0.997),
    ha="center",
    fontsize=11,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.5
    )
)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    OUT_DIR / "random_forest_model_selection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ================================================================
# Random Forest Classification Report Graph
# ================================================================

rf_report = {
    "Metric": [
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Legitimate": [
        0.9857,
        0.9855,
        0.9856
    ],
    "Phishing": [
        0.9857,
        0.9758,
        0.9807
    ]
}

metrics = rf_report["Metric"]
legitimate = rf_report["Legitimate"]
phishing = rf_report["Phishing"]

x_report = np.arange(len(metrics))
width_report = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars_legitimate = ax.bar(
    x_report - width_report / 2,
    legitimate,
    width_report,
    label="Legitimate"
)

bars_phishing = ax.bar(
    x_report + width_report / 2,
    phishing,
    width_report,
    label="Phishing"
)

# Add values above bars
for bars in [bars_legitimate, bars_phishing]:
    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

# Labels
ax.set_xlabel(
    "Performance Metric",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Performance Score",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Random Forest Classification Report",
    fontsize=15,
    fontweight="bold"
)

ax.set_xticks(x_report)
ax.set_xticklabels(metrics)

# Keep the same scale as the model comparison graph
ax.set_ylim(0.90, 1.00)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=2
)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    OUT_DIR / "random_forest_classification_report.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ================================================================
# 1. RANDOM FOREST EXTENDED PERFORMANCE METRICS
# ================================================================

rf_metrics = [
    "F1-Score",
    "Balanced Accuracy",
    "MCC",
    "Cohen's Kappa",
    "ROC-AUC",
    "PR-AUC"
]

rf_scores = [
    0.9807,
    0.9807,
    0.9613,
    0.9612,
    0.9971,
    0.9975
]

x_rf = np.arange(len(rf_metrics))

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.bar(
    x_rf,
    rf_scores,
    edgecolor="black",
    linewidth=1.5
)

# Add values
for bar, value in zip(bars, rf_scores):
    ax.annotate(
        f"{value:.4f}",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            value
        ),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

ax.set_title(
    "Random Forest Performance Metrics",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Performance Metric",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Performance Score",
    fontsize=12,
    fontweight="bold"
)

ax.set_xticks(x_rf)
ax.set_xticklabels(
    rf_metrics,
    rotation=20,
    ha="right"
)

ax.set_ylim(0.90, 1.00)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "random_forest_extended_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 2. CLASSIFICATION ERROR COMPARISON
# ================================================================

classification_error = [
    1 - accuracy[0],
    1 - accuracy[1],
    1 - accuracy[2],
    1 - accuracy[3]
]

# Convert to percentages
classification_error_percent = [
    error * 100
    for error in classification_error
]

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.bar(
    models,
    classification_error_percent,
    edgecolor="black",
    linewidth=1.5
)

# Highlight Random Forest
bars[3].set_linewidth(3)

# Add percentage values
for i, (bar, value) in enumerate(
    zip(bars, classification_error_percent)
):
    ax.annotate(
        f"{value:.2f}%",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            value
        ),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

ax.set_title(
    "Classification Error Comparison",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Machine Learning Model",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Classification Error (%)",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylim(
    0,
    max(classification_error_percent) + 1
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "classification_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



