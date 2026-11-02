import os
import uuid
import numpy as np
import cv2
from PIL import Image
import config
from preprocessing import load_and_preprocess_image, extract_anatomical_lung_mask
from model_utils import predict_detection

def get_suspicious_mask(
    display_image_rgb: np.ndarray, 
    heatmap_array: np.ndarray = None, 
    threshold: float = 0.35
) -> np.ndarray:
    """
    Derives an accurate, feathered mask of the suspicious lesion region inside the lungs.
    Combines Grad-CAM attention peaks with anatomical focal hyperdense lesion detection.
    Returns 2D float32 mask in range [0.0, 1.0].
    """
    h, w = display_image_rgb.shape[:2]
    gray = cv2.cvtColor(display_image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lung_mask = extract_anatomical_lung_mask(gray)

    # Detect high-density focal clusters inside the anatomical lung field
    focal_lesions = lung_mask & (gray > 0.36)

    # Incorporate Grad-CAM attention peak if available
    if heatmap_array is not None and np.max(heatmap_array) > 0.15:
        if heatmap_array.shape[:2] != (h, w):
            heatmap_resized = cv2.resize(heatmap_array, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap_array

        max_v = np.max(heatmap_resized)
        cutoff = max(threshold, max_v * 0.45)
        # Focus on heatmap activation strictly within the lung cavities
        cam_active = (heatmap_resized >= cutoff) & lung_mask
        focal_lesions = focal_lesions | cam_active

    if np.any(focal_lesions):
        # Dilate mask by 9 pixels to encompass entire lesion margin and spiculation boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(focal_lesions.astype(np.uint8), kernel)
        feathered = cv2.GaussianBlur(dilated.astype(np.float32), (15, 15), 0)
        max_f = np.max(feathered)
        if max_f > 0:
            feathered = np.clip(feathered / max_f, 0.0, 1.0)
        return feathered

    # Fallback to central lung nodule position if completely diffuse
    y_grid, x_grid = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    circ = (((x_grid - (cx - w * 0.15))**2 + (y_grid - (cy - h * 0.05))**2) <= (w * 0.08)**2).astype(np.float32)
    return cv2.GaussianBlur(circ, (21, 21), 0)


def apply_counterfactual_transformation(
    display_image_rgb: np.ndarray,
    mask: np.ndarray,
    mode: str,
    strength_val: float
) -> np.ndarray:
    """
    Applies transparent counterfactual modification to remove or attenuate the suspicious lesion.
    Supports:
      - 'Mask suspicious region': Excises cancer nodule and inpaints with healthy aerated lung parenchyma.
      - 'Blur suspicious region': Heavily blurs lesion to dissolve spiculation and high frequencies.
      - 'Reduce intensity in suspicious region': Attenuates lesion density to baseline healthy lung levels.
    """
    gray = cv2.cvtColor(display_image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lung_mask = extract_anatomical_lung_mask(gray)
    h, w = display_image_rgb.shape[:2]

    # Calculate median healthy aerated lung color
    healthy_pixels = display_image_rgb[lung_mask & (gray < 0.32)]
    if len(healthy_pixels) > 0:
        healthy_color = np.median(healthy_pixels, axis=0).astype(np.float32)
    else:
        healthy_color = np.array([36.0, 36.0, 36.0], dtype=np.float32)

    mask_3d = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    effective_alpha = np.clip(mask_3d * (0.80 + 0.20 * strength_val), 0.0, 1.0)

    if "mask" in mode.lower():
        # Clean excision: Inpaint using Telea algorithm from surrounding healthy parenchyma
        inpaint_mask = (mask > 0.08).astype(np.uint8) * 255
        if np.any(inpaint_mask):
            inpainted = cv2.inpaint(display_image_rgb, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA).astype(np.float32)
        else:
            inpainted = display_image_rgb.astype(np.float32)

        # Smoothly blend inpainted texture with healthy baseline color to fully remove lesion
        clean_target = (1.0 - (mask_3d * 0.90)) * inpainted + (mask_3d * 0.90) * healthy_color
        modified = (1.0 - effective_alpha) * display_image_rgb.astype(np.float32) + effective_alpha * clean_target

    elif "blur" in mode.lower():
        # Heavy Gaussian blur to dissolve spiculated nodule contours
        kernel_size = int(21 + 40 * strength_val)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(display_image_rgb, (kernel_size, kernel_size), 0).astype(np.float32)
        modified = (1.0 - effective_alpha) * display_image_rgb.astype(np.float32) + effective_alpha * blurred

    elif "reduce" in mode.lower() or "intensity" in mode.lower():
        # Attenuate lesion radiodensity down toward healthy aerated lung
        clean_target = healthy_color
        modified = (1.0 - effective_alpha) * display_image_rgb.astype(np.float32) + effective_alpha * clean_target

    else:
        # Default to clean inpainting
        inpaint_mask = (mask > 0.08).astype(np.uint8) * 255
        inpainted = cv2.inpaint(display_image_rgb, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA).astype(np.float32)
        clean_target = (1.0 - mask_3d) * inpainted + mask_3d * healthy_color
        modified = (1.0 - effective_alpha) * display_image_rgb.astype(np.float32) + effective_alpha * clean_target

    return np.clip(modified, 0, 255).astype(np.uint8)


def generate_explanation(
    mode: str,
    strength_label: str,
    orig_class: str,
    orig_cancer_prob: float,
    cf_class: str,
    cf_cancer_prob: float,
    prediction_changed: bool
) -> str:
    """
    Constructs a clear, academic explainable-AI narrative describing
    the causal effect of the counterfactual intervention.
    """
    drop = orig_cancer_prob - cf_cancer_prob
    drop_pct = round(abs(drop) * 100.0, 1)
    orig_pct = round(orig_cancer_prob * 100.0, 1)
    cf_pct = round(cf_cancer_prob * 100.0, 1)

    if orig_class == "Cancer" and cf_class == "Normal":
        return (
            f"What-If Analysis Confirmed Causal Impact: When the suspicious region identified by Grad-CAM "
            f"was modified using '{mode}' at '{strength_label}' strength, the model's Cancer probability "
            f"dropped significantly by {drop_pct}% (from {orig_pct}% down to {cf_pct}%). "
            f"This induced a decision flip from {orig_class} to {cf_class}. "
            f"This rigorously demonstrates that the focal lesion was the decisive causal feature "
            f"driving the network's original malignancy prediction."
        )
    elif orig_class == "Cancer" and cf_class == "Cancer":
        if drop > 0:
            return (
                f"What-If Sensitivity Observed: Modifying the suspicious region using '{mode}' "
                f"at '{strength_label}' strength reduced Cancer probability by {drop_pct}% "
                f"(from {orig_pct}% to {cf_pct}%). While the prediction remained {cf_class}, "
                f"the model showed clear sensitivity to the altered opacity. A higher intervention strength "
                f"or wider excision margin may be evaluated."
            )
        else:
            return (
                f"What-If Resilience Observed: The '{mode}' intervention at '{strength_label}' strength "
                f"yielded minimal change in Cancer probability ({orig_pct}% → {cf_pct}%). "
                f"The network continues to recognize secondary architectural patterns or residual opacities."
            )
    else:
        # Original was Normal
        if cf_cancer_prob > orig_cancer_prob:
            return (
                f"What-If Perturbation Effect: The original image was classified as {orig_class} ({round(100.0-orig_pct, 1)}% Normal). "
                f"Applying '{mode}' adjusted the texture, shifting Cancer probability slightly from {orig_pct}% to {cf_pct}%. "
                f"The classification remained {cf_class}."
            )
        else:
            return (
                f"What-If Consistency: The original image was classified as {orig_class} ({round(100.0-orig_pct, 1)}% Normal). "
                f"Applying '{mode}' at '{strength_label}' strength retained high confidence in {cf_class} ({round(100.0-cf_pct, 1)}% Normal), "
                f"confirming baseline absence of focal malignancy triggers."
            )


def run_whatif_analysis(
    detection_model,
    original_image_path: str,
    display_image_rgb: np.ndarray,
    heatmap_array: np.ndarray = None,
    mode: str = config.DEFAULT_MODE,
    strength_label: str = config.DEFAULT_STRENGTH,
    output_folder: str = None
) -> dict:
    """
    Executes complete Counterfactual What-If pipeline:
      1. Isolates suspicious lesion region using Grad-CAM heatmap.
      2. Modifies suspicious region using selected mode and strength.
      3. Saves modified counterfactual image.
      4. Re-preprocesses and runs modified image through the SAME CNN model.
      5. Compares original vs counterfactual probabilities.
      6. Formulates explainability narrative.
    """
    if output_folder is None:
        output_folder = config.GENERATED_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # 1. Run inference on original image
    orig_batch, _ = load_and_preprocess_image(original_image_path)
    orig_pred = predict_detection(detection_model, orig_batch)

    # 2. Get strength numeric factor
    strength_val = config.COUNTERFACTUAL_STRENGTHS.get(strength_label, 0.65)

    # 3. Derive mask for suspicious region
    mask = get_suspicious_mask(display_image_rgb, heatmap_array=heatmap_array)

    # 4. Apply Counterfactual modification
    modified_rgb = apply_counterfactual_transformation(display_image_rgb, mask, mode, strength_val)

    # 5. Save counterfactual image
    unique_id = uuid.uuid4().hex[:8]
    cf_filename = f"cf_{unique_id}.png"
    cf_path = os.path.join(output_folder, cf_filename)
    Image.fromarray(modified_rgb).save(cf_path, format="PNG")

    # 6. Preprocess counterfactual image and run through SAME model
    cf_batch, _ = load_and_preprocess_image(cf_path)
    cf_pred = predict_detection(detection_model, cf_batch)

    # 7. Compute probability shifts and verdict
    orig_cancer_p = orig_pred["cancer_probability"]
    cf_cancer_p = cf_pred["cancer_probability"]
    confidence_change = round(cf_cancer_p - orig_cancer_p, 4)
    abs_change = round(abs(confidence_change), 4)
    prediction_changed = (orig_pred["predicted_class"] != cf_pred["predicted_class"])

    explanation = generate_explanation(
        mode=mode,
        strength_label=strength_label,
        orig_class=orig_pred["predicted_class"],
        orig_cancer_prob=orig_cancer_p,
        cf_class=cf_pred["predicted_class"],
        cf_cancer_prob=cf_cancer_p,
        prediction_changed=prediction_changed
    )

    return {
        "success": True,
        "mode": mode,
        "strength_label": strength_label,
        "strength_val": strength_val,
        "cf_filename": cf_filename,
        "cf_path": cf_path,
        # Original Metrics
        "original_class": orig_pred["predicted_class"],
        "original_confidence": orig_pred["confidence"],
        "original_confidence_pct": orig_pred["confidence_pct"],
        "original_cancer_probability": orig_cancer_p,
        "original_cancer_probability_pct": orig_pred["cancer_probability_pct"],
        "original_normal_probability": orig_pred["normal_probability"],
        "original_normal_probability_pct": orig_pred["normal_probability_pct"],
        "original_probabilities": orig_pred["probabilities"],
        # Counterfactual Metrics
        "counterfactual_class": cf_pred["predicted_class"],
        "counterfactual_confidence": cf_pred["confidence"],
        "counterfactual_confidence_pct": cf_pred["confidence_pct"],
        "counterfactual_cancer_probability": cf_cancer_p,
        "counterfactual_cancer_probability_pct": cf_pred["cancer_probability_pct"],
        "counterfactual_normal_probability": cf_pred["normal_probability"],
        "counterfactual_normal_probability_pct": cf_pred["normal_probability_pct"],
        "counterfactual_probabilities": cf_pred["probabilities"],
        # Shift metrics
        "confidence_change": confidence_change,
        "confidence_change_pct": round(confidence_change * 100.0, 1),
        "abs_change": abs_change,
        "abs_change_pct": round(abs_change * 100.0, 1),
        "prediction_changed": prediction_changed,
        "prediction_changed_str": "YES" if prediction_changed else "NO",
        "explanation": explanation
    }
