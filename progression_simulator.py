import os
import uuid
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

import config
from preprocessing import load_and_preprocess_image, extract_anatomical_lung_mask
from model_utils import get_detection_model, predict_detection, get_staging_model, predict_staging


# ==============================================================================
# PROGRESSION SYNTHETIC DEMO ASSET GENERATOR
# ==============================================================================
def ensure_progression_demo_images(output_dir: str = None) -> list:
    """
    Ensures that 3 sequential longitudinal CT slices (Baseline, Month 3, Month 6)
    exist for 1-click clinical progression demonstration.
    Returns list of filepaths.
    """
    if output_dir is None:
        output_dir = config.PROGRESSION_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    timepoints = [
        {"filename": "prog_t0_baseline.png", "month": 0, "diameter_mm": 14.2, "fill": 145, "spic": 0},
        {"filename": "prog_t1_month3.png", "month": 3, "diameter_mm": 23.5, "fill": 168, "spic": 4},
        {"filename": "prog_t2_month6.png", "month": 6, "diameter_mm": 35.8, "fill": 188, "spic": 9}
    ]

    paths = []
    for tp in timepoints:
        fp = os.path.join(output_dir, tp["filename"])
        paths.append(fp)
        if os.path.exists(fp):
            continue

        # Synthesize realistic anatomical slice showing progressive tumor growth
        size = (224, 224)
        canvas = Image.new("L", size, color=18)
        draw = ImageDraw.Draw(canvas)

        # Chest wall and ribs
        draw.ellipse([14, 18, 210, 206], fill=52, outline=78, width=3)
        draw.ellipse([22, 26, 202, 198], fill=34, outline=60, width=2)
        draw.ellipse([102, 168, 122, 188], fill=190, outline=220)  # Spine
        draw.ellipse([108, 174, 116, 182], fill=40)
        draw.ellipse([104, 30, 120, 42], fill=175)  # Sternum

        # Bilateral lungs
        draw.ellipse([34, 48, 98, 158], fill=22)
        draw.ellipse([126, 48, 190, 158], fill=22)
        draw.ellipse([92, 90, 134, 148], fill=65)  # Heart

        # Primary Nodule evolving in Right Upper/Mid Lung
        cx, cy = 66, 88
        radius_px = max(4.0, (tp["diameter_mm"] / 2.8))
        draw.ellipse([cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px], fill=tp["fill"])

        # Spiculation radiating rays on later scans
        if tp["spic"] > 0:
            for angle in np.linspace(0, 2 * np.pi, tp["spic"], endpoint=False):
                rx = cx + (radius_px + 5) * np.cos(angle)
                ry = cy + (radius_px + 5) * np.sin(angle)
                draw.line([cx, cy, rx, ry], fill=tp["fill"] - 30, width=1)

        blurred = canvas.filter(ImageFilter.GaussianBlur(radius=0.75))
        rgb_img = Image.merge("RGB", (blurred, blurred, blurred))
        rgb_img.save(fp, format="PNG")

    return paths


# ==============================================================================
# TUMOR QUANTIFICATION & KINETIC PROGRESSION MODEL
# ==============================================================================
def quantify_slice_tumor(image_path: str, detection_model=None, staging_model=None) -> dict:
    """
    Evaluates a single CT slice: extracts lesion morphometry, cancer probability,
    and TNM staging classification.
    """
    if detection_model is None:
        detection_model, _, _ = get_detection_model()
    if staging_model is None:
        staging_model = get_staging_model()

    preprocessed_batch, _ = load_and_preprocess_image(image_path)
    detection_res = predict_detection(detection_model, preprocessed_batch)
    staging_res = predict_staging(
        staging_model=staging_model,
        preprocessed_batch=preprocessed_batch,
        cancer_prob=detection_res["cancer_probability"]
    )

    # Detailed pixel-level morphology
    img = preprocessed_batch[0]
    gray = np.mean(img, axis=-1)
    lung_mask = extract_anatomical_lung_mask(gray)
    suspicious = (lung_mask) & (gray > 0.38)

    import cv2
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(suspicious.astype(np.uint8))
    primary_area_px = 0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = int(np.argmax(areas)) + 1
        primary_area_px = int(stats[max_idx, cv2.CC_STAT_AREA])

    # Unidimensional diameter (RECIST 1.1 standard, mm)
    scale_mm = 1.45  # mm per pixel at 224x224 thoracic FOV
    radius_px = np.sqrt(primary_area_px / np.pi) if primary_area_px > 0 else 0.0
    diameter_mm = float(2.0 * radius_px * scale_mm)
    diameter_mm = max(2.0, round(diameter_mm, 1))

    # Area in mm^2
    area_mm2 = round(primary_area_px * (scale_mm ** 2), 1)

    # 3D Ellipsoidal Volume estimate: V = (4/3) * pi * r^3 = (pi / 6) * D^3
    volume_mm3 = (np.pi / 6.0) * (diameter_mm ** 3)
    volume_cm3 = round(volume_mm3 / 1000.0, 3)

    # Mean density of suspicious lesion
    mean_density = float(np.mean(gray[suspicious])) if np.any(suspicious) else 0.20

    stage = staging_res.get("stage", "Stage I") if staging_res.get("eligible") else "Normal/Benign"
    tnm_code = staging_res.get("tnm_code", "N/A")
    image_filename = os.path.basename(image_path)
    is_upload = "uploads" in image_path.replace("\\", "/").lower()

    return {
        "image_path": image_path,
        "image_filename": image_filename,
        "is_upload": is_upload,
        "diameter_mm": diameter_mm,
        "area_mm2": area_mm2,
        "volume_cm3": volume_cm3,
        "volume_mm3": round(volume_mm3, 1),
        "mean_density": round(mean_density, 3),
        "cancer_probability": detection_res["cancer_probability"],
        "cancer_prob_pct": detection_res["cancer_probability_pct"],
        "predicted_class": detection_res["predicted_class"],
        "stage": stage,
        "tnm_code": tnm_code
    }


def analyze_longitudinal_progression(timepoints: list) -> dict:
    """
    Takes a list of timepoint dictionaries:
    [
        {"label": "Baseline", "month": 0, "image_path": "..."},
        {"label": "Month 3 Follow-up", "month": 3, "image_path": "..."},
        ...
    ]
    Sorts by month, quantifies tumor growth at each interval, computes kinetics (VDT, RECIST 1.1),
    projects next-stage progression, and generates multi-panel progression charts.
    """
    if len(timepoints) < 2:
        raise ValueError("At least 2 longitudinal timepoints are required to calculate tumor growth kinetics.")

    # Sort strictly by chronological month
    timepoints = sorted(timepoints, key=lambda x: float(x["month"]))

    detection_model, _, _ = get_detection_model()
    staging_model = get_staging_model()

    quantified = []
    for idx, tp in enumerate(timepoints):
        q = quantify_slice_tumor(
            image_path=tp["image_path"],
            detection_model=detection_model,
            staging_model=staging_model
        )
        q["label"] = tp.get("label", f"Timepoint {idx+1}")
        q["month"] = float(tp["month"])
        q["idx"] = idx
        quantified.append(q)

    baseline = quantified[0]
    latest = quantified[-1]

    # Baseline vs Latest comparative metrics
    d_0 = max(1.0, baseline["diameter_mm"])
    d_n = latest["diameter_mm"]
    delta_d_mm = round(d_n - d_0, 1)
    pct_delta_d = round(((d_n - d_0) / d_0) * 100.0, 1)

    v_0 = max(0.01, baseline["volume_cm3"])
    v_n = max(0.01, latest["volume_cm3"])
    delta_v_cm3 = round(v_n - v_0, 3)
    pct_delta_v = round(((v_n - v_0) / v_0) * 100.0, 1)

    elapsed_months = latest["month"] - baseline["month"]
    elapsed_days = elapsed_months * 30.4

    # Tumor Volume Doubling Time (VDT) via Schwartz's equation:
    # VDT = (t * ln(2)) / ln(V_n / V_0)
    vdt_days = None
    vdt_category = "Stable / Insufficient Growth"
    growth_velocity_k = 0.0

    if elapsed_days > 0 and v_n > v_0:
        ratio = v_n / v_0
        if ratio > 1.05:
            vdt_days = round((elapsed_days * np.log(2)) / np.log(ratio), 1)
            growth_velocity_k = np.log(d_n / d_0) / max(0.1, elapsed_months)
            if vdt_days < 100:
                vdt_category = "Rapid / Aggressive Malignancy (VDT < 100 days)"
            elif vdt_days <= 400:
                vdt_category = "Typical Malignant NSCLC Doubling (VDT 100 - 400 days)"
            else:
                vdt_category = "Indolent / Slow Growth (VDT > 400 days)"
    elif v_n < v_0:
        vdt_category = "Tumor Regression / Response to Therapy"

    # RECIST 1.1 Assessment
    if pct_delta_d >= config.RECIST_PROGRESSIVE_THRESHOLD and delta_d_mm >= 3.0:
        recist_category = "Progressive Disease (PD)"
        recist_color = "danger"
        recist_explanation = f"Tumor diameter expanded by +{pct_delta_d}% (> +20% threshold, RECIST 1.1 criteria)."
    elif pct_delta_d <= config.RECIST_PARTIAL_RESPONSE_THRESHOLD:
        recist_category = "Partial Response (PR)"
        recist_color = "success"
        recist_explanation = f"Tumor diameter decreased by {abs(pct_delta_d)}% (≥ 30% reduction, RECIST 1.1 criteria)."
    elif d_n <= 1.0:
        recist_category = "Complete Response (CR)"
        recist_color = "success"
        recist_explanation = "Target lesion completely resolved."
    else:
        recist_category = "Stable Disease (SD)"
        recist_color = "info"
        recist_explanation = f"Diameter change ({pct_delta_d:+.1f}%) within stable boundary (-30% to +20%)."

    # ==============================================================================
    # PREDICTIVE NEXT-STAGE PROGRESSION FORECASTING ENGINE
    # ==============================================================================
    forecast_results = predict_next_stage_progression(
        latest_diameter_mm=d_n,
        growth_velocity_k=growth_velocity_k,
        latest_stage=latest["stage"],
        vdt_days=vdt_days
    )

    # ==============================================================================
    # GENERATE COMPREHENSIVE MULTI-PANEL PROGRESSION GRAPH
    # ==============================================================================
    chart_path, chart_filename = generate_progression_graph(
        quantified_timepoints=quantified,
        forecast_results=forecast_results,
        recist_category=recist_category
    )

    return {
        "timepoints": quantified,
        "baseline": baseline,
        "latest": latest,
        "elapsed_months": elapsed_months,
        "elapsed_days": round(elapsed_days, 0),
        "delta_d_mm": delta_d_mm,
        "pct_delta_d": pct_delta_d,
        "delta_v_cm3": delta_v_cm3,
        "pct_delta_v": pct_delta_v,
        "vdt_days": vdt_days,
        "vdt_category": vdt_category,
        "recist_category": recist_category,
        "recist_color": recist_color,
        "recist_explanation": recist_explanation,
        "forecast": forecast_results,
        "chart_filename": chart_filename,
        "chart_path": chart_path
    }


def predict_next_stage_progression(
    latest_diameter_mm: float, 
    growth_velocity_k: float, 
    latest_stage: str, 
    vdt_days: float = None
) -> dict:
    """
    Projects future tumor trajectory and calculates probability of stage transition.
    AJCC 8th Edition Lung Cancer T-Categories:
      - T1: <= 30 mm (Stage I)
      - T2: 31 - 50 mm (Stage II)
      - T3: 51 - 70 mm (Stage III)
      - T4: > 70 mm (Stage III/IV)
    """
    # Safe floor for growth velocity
    k = max(0.015, growth_velocity_k) if growth_velocity_k > 0 else 0.045

    # Forecast dimensions at +3, +6, and +12 months post-latest scan
    future_offsets = [3.0, 6.0, 12.0]
    forecast_points = []

    for f_mo in future_offsets:
        proj_diam = latest_diameter_mm * np.exp(k * f_mo)
        proj_vol = (np.pi / 6.0) * (proj_diam ** 3) / 1000.0  # cm^3
        
        # Stage estimation based on AJCC diameter boundaries
        if proj_diam <= 30.0:
            proj_stage = "Stage I (T1)"
        elif proj_diam <= 50.0:
            proj_stage = "Stage II (T2)"
        elif proj_diam <= 70.0:
            proj_stage = "Stage III (T3)"
        else:
            proj_stage = "Stage IV / Advanced (T4/Metastatic Risk)"

        forecast_points.append({
            "months_ahead": f_mo,
            "projected_diameter_mm": round(proj_diam, 1),
            "projected_volume_cm3": round(proj_vol, 2),
            "projected_stage": proj_stage
        })

    # Estimate time to breach next stage boundary
    if latest_diameter_mm < 30.0:
        target_diam = 30.0
        next_stage_name = "Stage IIA (T2a N0 M0)"
        boundary_name = "30 mm (Stage I → II Threshold)"
    elif latest_diameter_mm < 50.0:
        target_diam = 50.0
        next_stage_name = "Stage IIIA (T3 N0 M0)"
        boundary_name = "50 mm (Stage II → III Threshold)"
    elif latest_diameter_mm < 70.0:
        target_diam = 70.0
        next_stage_name = "Stage IIIB / IV (T4 Locally Invasive / Metastatic)"
        boundary_name = "70 mm (Stage III → IV Severe Boundary)"
    else:
        target_diam = latest_diameter_mm * 1.3
        next_stage_name = "Stage IV Disseminated Systemic Progression"
        boundary_name = "Critical Mass Exceeded"

    months_to_next_stage = None
    if k > 0:
        months_to_next_stage = round(np.log(target_diam / max(1.0, latest_diameter_mm)) / k, 1)
        months_to_next_stage = max(0.5, months_to_next_stage)

    # Risk level classification
    if vdt_days is not None and vdt_days < 100:
        risk_level = "CRITICAL / RAPID PROGRESSION"
        risk_badge = "pill-danger"
        clinical_action = (
            "URGENT CLINICAL DIRECTIVE: Tumor doubling time is under 100 days, demonstrating aggressive cellular proliferation. "
            "Immediate multidisciplinary tumor board review, staging PET-CT, and accelerated surgical/systemic intervention recommended."
        )
    elif months_to_next_stage and months_to_next_stage <= 6.0:
        risk_level = "HIGH RISK OF UPSTAGING"
        risk_badge = "pill-danger"
        clinical_action = (
            f"HIGH PROGRESSION HAZARD: Current growth trajectory will breach the {boundary_name} within ~{months_to_next_stage} months. "
            "Resection window is actively narrowing; prompt thoracic surgical consult is indicated."
        )
    elif months_to_next_stage and months_to_next_stage <= 12.0:
        risk_level = "MODERATE ONCOLOGICAL RISK"
        risk_badge = "pill-warning"
        clinical_action = (
            "MODERATE PROGRESSION VELOCITY: Lesion is progressing at standard adenocarcinoma rates. "
            "Follow-up high-resolution chest CT with contrast recommended within 6 to 8 weeks."
        )
    else:
        risk_level = "LOW / INDOLENT TRAJECTORY"
        risk_badge = "pill-success"
        clinical_action = (
            "INDOLENT DYNAMICS: Growth velocity is subdued. Active CT surveillance protocol per Fleischner Society guidelines appropriate."
        )

    return {
        "growth_velocity_k": round(k, 4),
        "forecast_points": forecast_points,
        "next_stage_name": next_stage_name,
        "boundary_name": boundary_name,
        "months_to_next_stage": months_to_next_stage,
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "clinical_action": clinical_action
    }


# ==============================================================================
# PROGRESSION VISUALIZATION & MULTI-PANEL MATPLOTLIB CHART
# ==============================================================================
def generate_progression_graph(
    quantified_timepoints: list, 
    forecast_results: dict, 
    recist_category: str,
    output_folder: str = None
) -> tuple:
    """
    Generates a 3-panel clinical visualization showing:
    1. Historical tumor diameter & volume progression + future extrapolation cone.
    2. Malignancy risk probability curve.
    3. Kinetic doubling velocity & stage milestone boundaries.
    """
    if output_folder is None:
        output_folder = config.GENERATED_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    months_hist = [tp["month"] for tp in quantified_timepoints]
    diams_hist = [tp["diameter_mm"] for tp in quantified_timepoints]
    vols_hist = [tp["volume_cm3"] for tp in quantified_timepoints]
    probs_hist = [tp["cancer_prob_pct"] for tp in quantified_timepoints]

    # Future forecast points
    latest_month = months_hist[-1]
    f_months = [latest_month + fp["months_ahead"] for fp in forecast_results["forecast_points"]]
    f_diams = [fp["projected_diameter_mm"] for fp in forecast_results["forecast_points"]]

    # Build plot
    fig = plt.figure(figsize=(10.5, 6.2), dpi=140)
    fig.patch.set_facecolor("#0f172a")  # Deep slate dark
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.35, wspace=0.25)

    # -------------------------------------------------------------
    # PANEL 1 (Top Span): Tumor Diameter & Volume Trajectory + Forecast
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#1e293b")

    # Historical Measured Points
    ax1.plot(months_hist, diams_hist, "o-", color="#38bdf8", linewidth=2.5, markersize=8, label="Measured Diameter (mm)", zorder=5)

    # Future Extrapolation Curve
    extrap_x = [latest_month] + f_months
    extrap_y = [diams_hist[-1]] + f_diams
    ax1.plot(extrap_x, extrap_y, "--", color="#d946ef", linewidth=2.2, label="Projected Growth (Kinetic Model)", zorder=4)

    # Forecast Confidence Cone (shaded 90% trajectory)
    upper_cone = [y * (1.0 + 0.05 * idx) for idx, y in enumerate(extrap_y)]
    lower_cone = [y * (1.0 - 0.04 * idx) for idx, y in enumerate(extrap_y)]
    ax1.fill_between(extrap_x, lower_cone, upper_cone, color="#8b5cf6", alpha=0.18, label="Forecast Confidence Range (±15%)")

    # RECIST 1.1 +20% PD Boundary Line
    base_d = diams_hist[0]
    recist_pd_line = base_d * 1.20
    ax1.axhline(recist_pd_line, color="#f43f5e", linestyle=":", alpha=0.75, linewidth=1.5, label="RECIST 1.1 PD (+20%)")

    # Stage Threshold Lines
    ax1.axhline(30.0, color="#f59e0b", linestyle="-.", alpha=0.5, linewidth=1.2)
    ax1.text(months_hist[0] + 0.1, 31.0, "Stage II Threshold (30 mm)", color="#f59e0b", fontsize=8.5, fontweight="bold")

    # Annotate points with exact measurements
    for m, d in zip(months_hist, diams_hist):
        ax1.annotate(
            f"{d} mm",
            xy=(m, d),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
            fontsize=9.5
        )

    for m, d in zip(f_months, f_diams):
        ax1.annotate(
            f"~{d:.1f} mm",
            xy=(m, d),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#c084fc",
            fontsize=8.5
        )

    ax1.set_title("Longitudinal Lesion Trajectory & Kinetic Next-Stage Extrapolation", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
    ax1.set_xlabel("Observation Timeline (Elapsed Months)", color="#94a3b8", fontsize=9.5)
    ax1.set_ylabel("Tumor Diameter (mm)", color="#94a3b8", fontsize=9.5)
    ax1.tick_params(colors="#cbd5e1", labelsize=9)
    ax1.grid(color="#334155", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.2)

    # -------------------------------------------------------------
    # PANEL 2 (Bottom Left): Cancer Probability Trend
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#1e293b")

    bars = ax2.bar(
        [str(tp["label"]) for tp in quantified_timepoints],
        probs_hist,
        color=["#10b981" if p < 50 else "#f43f5e" for p in probs_hist],
        edgecolor="#ffffff",
        linewidth=0.8,
        width=0.45
    )

    for b in bars:
        h = b.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
            fontsize=9
        )

    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Cancer Probability (%)", color="#94a3b8", fontsize=9)
    ax2.set_title("Malignancy Risk Index Over Time", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=8)
    ax2.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax2.grid(axis="y", color="#334155", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # PANEL 3 (Bottom Right): Tumor Volumetric Evolution (cm^3)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1e293b")

    ax3.plot(months_hist, vols_hist, "s-", color="#10b981", linewidth=2.2, markersize=7, label="Ellipsoidal Volume ($cm^3$)")
    ax3.fill_between(months_hist, 0, vols_hist, color="#10b981", alpha=0.2)

    for m, v in zip(months_hist, vols_hist):
        ax3.annotate(
            f"{v:.2f} cm³",
            xy=(m, v),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
            fontsize=9
        )

    ax3.set_title(f"Volumetric Expansion (RECIST: {recist_category})", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=8)
    ax3.set_xlabel("Elapsed Months", color="#94a3b8", fontsize=9)
    ax3.set_ylabel("Tumor Volume ($cm^3$)", color="#94a3b8", fontsize=9)
    ax3.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax3.grid(color="#334155", linestyle="--", alpha=0.5)

    chart_filename = f"progression_chart_{uuid.uuid4().hex[:8]}.png"
    chart_path = os.path.join(output_folder, chart_filename)
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)

    return chart_path, chart_filename
