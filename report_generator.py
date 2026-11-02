import os
import uuid
import datetime
import config

def build_report_payload(
    image_filename: str,
    detection_res: dict,
    gradcam_res: dict = None,
    cf_res: dict = None,
    staging_res: dict = None,
    similar_cases: list = None
) -> dict:
    """
    Consolidates analysis metrics into a formal academic report data structure,
    including Counterfactual explainability and Similar Case comparative insights.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_id = f"LUNG-XAI-{uuid.uuid4().hex[:8].upper()}"

    is_demo = detection_res.get("is_demo", True)
    model_status = "DEMO MODE (Educational Simulation)" if is_demo else "TRAINED CNN MODEL LOADED"

    report = {
        "report_id": report_id,
        "timestamp": timestamp,
        "project_title": config.PROJECT_TITLE,
        "academic_scope": config.ACADEMIC_YEAR,
        "model_status": model_status,
        "is_demo": is_demo,
        "demo_notice": config.DEMO_MODE_NOTICE if is_demo else None,
        "disclaimer": config.DISCLAIMER,
        "image_filename": image_filename,
        # Detection
        "predicted_class": detection_res.get("predicted_class", "N/A"),
        "confidence_pct": detection_res.get("confidence_pct", 0.0),
        "cancer_prob_pct": detection_res.get("cancer_probability_pct", 0.0),
        "normal_prob_pct": detection_res.get("normal_probability_pct", 0.0),
        # Grad-CAM
        "gradcam_status": "Computed & Verified" if (gradcam_res and gradcam_res.get("success")) else "Not Computed / Unavailable",
        "gradcam_heatmap_file": gradcam_res.get("heatmap_filename") if gradcam_res else None,
        "gradcam_overlay_file": gradcam_res.get("overlay_filename") if gradcam_res else None,
        # Counterfactual What-If
        "has_counterfactual": bool(cf_res and cf_res.get("success")),
        "cf_mode": cf_res.get("mode") if cf_res else "N/A",
        "cf_strength": cf_res.get("strength_label") if cf_res else "N/A",
        "cf_predicted_class": cf_res.get("counterfactual_class") if cf_res else "N/A",
        "cf_confidence_pct": cf_res.get("counterfactual_confidence_pct") if cf_res else "N/A",
        "cf_cancer_prob_pct": cf_res.get("counterfactual_cancer_probability_pct") if cf_res else "N/A",
        "confidence_change_pct": cf_res.get("confidence_change_pct") if cf_res else "N/A",
        "prediction_changed": cf_res.get("prediction_changed_str") if cf_res else "N/A",
        "cf_explanation": cf_res.get("explanation") if cf_res else "Counterfactual analysis has not been executed yet.",
        "cf_image_file": cf_res.get("cf_filename") if cf_res else None,
        # Staging
        "has_staging": bool(staging_res and staging_res.get("available")),
        "staging_stage": staging_res.get("stage") if (staging_res and staging_res.get("available")) else "Not Available",
        "staging_confidence": staging_res.get("confidence", "N/A") if staging_res else "N/A",
        "staging_tnm_code": staging_res.get("tnm_code", "N/A") if staging_res else "N/A",
        "staging_description": staging_res.get("tnm_description", "N/A") if staging_res else "N/A",
        "staging_clinical_criteria": staging_res.get("clinical_criteria", "N/A") if staging_res else "N/A",
        "staging_treatment": staging_res.get("treatment_strategy", "N/A") if staging_res else "N/A",
        "staging_probabilities": staging_res.get("probabilities", {}) if staging_res else {},
        "staging_diameter_mm": staging_res.get("lesion_diameter_mm", "N/A") if staging_res else "N/A",
        "staging_message": staging_res.get("message") if staging_res else "Staging analysis not performed.",
        # Similar Case Retrieval
        "similar_cases": similar_cases or []
    }

    return report


def save_downloadable_report(report_data: dict, output_folder: str = None) -> tuple:
    """
    Saves a formal academic text report file ready for download.
    Returns (report_filepath, report_filename).
    """
    if output_folder is None:
        output_folder = config.REPORTS_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    report_id = report_data["report_id"]
    filename = f"Report_{report_id}.txt"
    filepath = os.path.join(output_folder, filename)

    divider = "=" * 80
    sub_divider = "-" * 80

    lines = [
        divider,
        f"  {report_data['project_title'].upper()}",
        f"  ACADEMIC RESEARCH & EXPLAINABLE AI (XAI) LABORATORY REPORT",
        divider,
        f"Report ID        : {report_data['report_id']}",
        f"Generated At     : {report_data['timestamp']}",
        f"Target Image     : {report_data['image_filename']}",
        f"Model Operating  : {report_data['model_status']}",
        sub_divider,
        "SECTION 1: CNN DETECTION INFERENCE",
        sub_divider,
        f"Primary Prediction       : {report_data['predicted_class']}",
        f"Overall Model Confidence : {report_data['confidence_pct']}%",
        f"Cancer Class Probability : {report_data['cancer_prob_pct']}%",
        f"Normal Class Probability : {report_data['normal_prob_pct']}%",
        "",
        sub_divider,
        "SECTION 2: GRAD-CAM EXPLAINABILITY (VISUAL ATTENTION)",
        sub_divider,
        f"Grad-CAM Heatmap Status  : {report_data['gradcam_status']}",
        f"Heatmap Artifact         : {report_data.get('gradcam_heatmap_file', 'None')}",
        f"Blended Overlay Artifact : {report_data.get('gradcam_overlay_file', 'None')}",
        "Clinical Interpretation  : Heatmap highlights convolutional gradient peaks corresponding",
        "                           to dense opacities, focal consolidations, or nodular masses.",
        "",
        sub_divider,
        "SECTION 3: COUNTERFACTUAL WHAT-IF CAUSAL ANALYSIS (CORE NOVELTY)",
        sub_divider,
        f"Intervention Mode Applied : {report_data['cf_mode']}",
        f"Intervention Strength     : {report_data['cf_strength']}",
        f"Counterfactual Class      : {report_data['cf_predicted_class']}",
        f"Counterfactual Confidence : {report_data['cf_confidence_pct']}%",
        f"Cancer Probability Shift  : {report_data['cancer_prob_pct']}% -> {report_data['cf_cancer_prob_pct']}%",
        f"Confidence Delta (Change) : {report_data['confidence_change_pct']}%",
        f"Prediction Flipped/Changed: {report_data['prediction_changed']}",
        "",
        "EXPLAINABILITY VERDICT:",
        f"  {report_data['cf_explanation']}",
        "",
        sub_divider,
        "SECTION 4: CANCER STAGING (TNM CORRELATION)",
        sub_divider,
        f"Staging Evaluation : {report_data['staging_stage']}",
        f"TNM Classification : {report_data.get('staging_tnm_code', 'N/A')}",
        f"Staging Confidence : {report_data.get('staging_confidence', 'N/A')}%",
        f"Estimated Diameter : ~{report_data.get('staging_diameter_mm', 'N/A')} mm",
        f"Pathological Note  : {report_data.get('staging_description', 'N/A')}",
        f"Clinical Pathway   : {report_data.get('staging_treatment', 'N/A')}",
        f"Staging Status     : {report_data['staging_message']}",
        ""
    ]

    # Similar cases section
    similar_cases = report_data.get("similar_cases", [])
    if similar_cases:
        lines.extend([
            sub_divider,
            "SECTION 5: SIMILAR CASE RETRIEVAL & COMPARATIVE CLINICAL INSIGHTS",
            sub_divider,
            f"Retrieved {len(similar_cases)} most radiomically concordant reference cases from oncology dataset:"
        ])
        for idx, sc in enumerate(similar_cases, 1):
            lines.extend([
                "",
                f"  Case #{idx}: {sc['case_id']} (Match Score: {sc['match_pct']}%)",
                f"    Patient Demographics : {sc['patient_age']}yo {sc['patient_gender']}, {sc['smoking_history']}",
                f"    Confirmed Diagnosis  : {sc['diagnosis']} [{sc['stage']}]",
                f"    Biomarkers & Genes   : {sc.get('biomarkers', 'N/A')}",
                f"    Primary Treatment    : {sc['treatment_regimen']['primary']}",
                f"    Systemic Therapy     : {sc['treatment_regimen']['systemic']}",
                f"    Treatment Response   : {sc['treatment_outcomes']['response']}",
                f"    Survival Outcome     : {sc['treatment_outcomes']['survival_metrics']}",
                f"    Physician Insight    : {sc['physician_insights']}"
            ])
        lines.append("")

    lines.extend([
        sub_divider,
        "SECTION 6: REGULATORY & MEDICAL ETHICS DISCLAIMER",
        sub_divider,
        f"{report_data['disclaimer']}",
        "",
        divider,
        "END OF ACADEMIC EVALUATION REPORT",
        divider
    ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath, filename
