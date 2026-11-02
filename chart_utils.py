import os
import uuid
import matplotlib
matplotlib.use("Agg")  # Headless backend for web server
import matplotlib.pyplot as plt
import numpy as np
import config

def generate_comparison_chart(
    original_probs: dict,
    counterfactual_probs: dict,
    prediction_changed: bool = False,
    output_folder: str = None
) -> tuple:
    """
    Generates a clear side-by-side bar chart comparing Original vs Counterfactual probabilities.
    original_probs: {"Normal": float, "Cancer": float}
    counterfactual_probs: {"Normal": float, "Cancer": float}
    Returns (chart_path, chart_filename).
    """
    if output_folder is None:
        output_folder = config.GENERATED_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    classes = ["Normal", "Cancer"]
    orig_vals = [original_probs.get(c, 0.0) * 100.0 for c in classes]
    cf_vals = [counterfactual_probs.get(c, 0.0) * 100.0 for c in classes]

    x = np.arange(len(classes))
    bar_width = 0.32

    # Set up stylish plot
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    fig.patch.set_facecolor("#0f172a")  # Deep slate dark background
    ax.set_facecolor("#1e293b")

    # Bar colors
    # Original: solid cyan/coral
    # Counterfactual: bright emerald/orange
    bars1 = ax.bar(
        x - bar_width / 2, 
        orig_vals, 
        bar_width, 
        label="Original Scan", 
        color="#38bdf8", 
        edgecolor="#0284c7", 
        linewidth=1.2,
        zorder=3
    )
    bars2 = ax.bar(
        x + bar_width / 2, 
        cf_vals, 
        bar_width, 
        label="Counterfactual (What-If)", 
        color="#f43f5e" if not prediction_changed else "#10b981", 
        edgecolor="#ffffff", 
        linewidth=1.2,
        zorder=3
    )

    # Gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8", zorder=0)

    # Value labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", 
            va="bottom",
            color="#e2e8f0",
            fontweight="bold",
            fontsize=9.5
        )

    for bar in bars2:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", 
            va="bottom",
            color="#f8fafc",
            fontweight="bold",
            fontsize=9.5
        )

    # Labels and Title
    ax.set_ylabel("Model Probability (%)", color="#94a3b8", fontsize=11, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)

    title_text = "What-If Causal Impact: Original vs Counterfactual Probabilities"
    ax.set_title(title_text, color="#f8fafc", fontsize=12, fontweight="bold", pad=12)

    # Style axes ticks and spines
    ax.tick_params(axis="y", colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Legend
    legend = ax.legend(
        loc="upper right", 
        framealpha=0.85, 
        facecolor="#0f172a", 
        edgecolor="#334155", 
        fontsize=9.5,
        labelcolor="#f8fafc"
    )

    # Annotation badge if prediction flipped
    verdict_text = "Decision Changed: YES" if prediction_changed else "Decision Changed: NO"
    badge_bg = "#065f46" if prediction_changed else "#1e293b"
    badge_border = "#10b981" if prediction_changed else "#64748b"
    ax.text(
        0.04, 0.92, 
        verdict_text, 
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="#f8fafc",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_bg, edgecolor=badge_border, alpha=0.9)
    )

    plt.tight_layout()

    unique_id = uuid.uuid4().hex[:8]
    chart_filename = f"chart_cf_{unique_id}.png"
    chart_path = os.path.join(output_folder, chart_filename)
    fig.savefig(chart_path, format="png", dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return chart_path, chart_filename
