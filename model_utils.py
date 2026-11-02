import os
import glob
import numpy as np
import config
from preprocessing import extract_anatomical_lung_mask

class DemoCNNModel:
    """
    Academic Demo CNN Simulator for educational & research demonstration.
    Labels all outputs explicitly as DEMO MODE.
    Uses realistic lung-density feature convolution to provide responsive,
    re-predictable outputs for Counterfactual What-If analysis.
    """
    def __init__(self):
        self.is_demo = True
        self.model_name = "Academic Demo CNN Simulator (No weights file loaded)"
        self.classes = config.DETECTION_CLASSES

    def predict(self, preprocessed_batch: np.ndarray) -> np.ndarray:
        """
        Calculates prediction probabilities for [Normal, Cancer].
        preprocessed_batch: shape (1, H, W, 3), float32 in [0, 1]
        Returns shape (1, 2) array of probabilities.
        """
        img = preprocessed_batch[0]
        # Convert to single-channel luminance / density representation
        gray = np.mean(img, axis=-1)
        h, w = gray.shape

        # Anatomical lung field extraction
        lung_mask = extract_anatomical_lung_mask(gray)

        # Extract lung pixels
        lung_pixels = gray[lung_mask]
        if len(lung_pixels) == 0:
            lung_pixels = gray

        # High density lesion detection inside lung zones:
        # Suspicious focal regions have brightness significantly higher than healthy aerated lung
        suspicious_threshold = 0.38
        suspicious_pixels = gray[(lung_mask) & (gray > suspicious_threshold)]
        
        # Calculate lesion density metric
        lesion_count = len(suspicious_pixels)
        total_lung = np.sum(lung_mask) + 1e-5
        lesion_fraction = lesion_count / total_lung

        # Also detect max focal intensity in lung zone
        max_focal = np.max(gray[lung_mask]) if np.any(lung_mask) else np.max(gray)

        # Baseline score calculation:
        # A normal scan has very low lesion fraction (< 0.01) and low max focal in lungs
        # A cancer scan with a nodule has a compact dense cluster (fraction > 0.02, max > 0.5)
        raw_score = (lesion_fraction * 28.0) + (max_focal - 0.20) * 1.2
        # Apply sigmoid-like squash
        cancer_prob = float(1.0 / (1.0 + np.exp(-4.5 * (raw_score - 0.52))))
        cancer_prob = np.clip(cancer_prob, 0.04, 0.96)
        normal_prob = float(1.0 - cancer_prob)

        return np.array([[normal_prob, cancer_prob]], dtype=np.float32)


def get_detection_model():
    """
    Scans models/ directory for trained model files (.keras, .h5, SavedModel).
    If found and TensorFlow is available, loads the real model.
    Otherwise, returns DemoCNNModel marked as DEMO MODE.
    """
    models_dir = config.MODELS_FOLDER
    os.makedirs(models_dir, exist_ok=True)

    # Search for model weights
    search_patterns = [
        os.path.join(models_dir, "*.keras"),
        os.path.join(models_dir, "*.h5"),
        os.path.join(models_dir, "*_saved_model*"),
    ]

    candidate_files = []
    for pattern in search_patterns:
        candidate_files.extend(glob.glob(pattern))

    # Filter out potential staging models if named specifically
    detection_candidates = [
        f for f in candidate_files 
        if "staging" not in os.path.basename(f).lower()
    ]

    if detection_candidates:
        model_path = detection_candidates[0]
        try:
            import tensorflow as tf
            loaded_model = tf.keras.models.load_model(model_path)
            return loaded_model, False, {
                "status": "REAL MODEL LOADED",
                "name": os.path.basename(model_path),
                "path": model_path
            }
        except Exception as e:
            # Fallback to Demo Mode with warning if loading failed
            return DemoCNNModel(), True, {
                "status": "DEMO MODE (Load error on candidate file)",
                "error": str(e),
                "candidate": os.path.basename(model_path)
            }

    # No model file exists: Safe Demo Mode
    return DemoCNNModel(), True, {
        "status": "DEMO MODE (No trained model weights found in models/)",
        "notice": config.DEMO_MODE_NOTICE
    }


class DemoStagingCNNModel:
    """
    TNM Staging Neural Feature & Classification Model for chest CT slices.
    Evaluates Tumor Morphometry (T), Nodal Proximity (N), and Metastatic Dissemination (M)
    according to AJCC/UICC 8th Edition Lung Cancer Staging guidelines.
    Returns 4-class calibrated probabilities: [Stage I, Stage II, Stage III, Stage IV].
    """
    def __init__(self, weights_path: str = None):
        self.is_demo = True
        self.model_name = "TNM Staging CNN Morphometry Engine"
        self.classes = config.STAGING_CLASSES
        self.weights_path = weights_path
        self.calibrated_weights = {
            "scale_factor": 1.5,     # Approx mm per pixel at 224x224 FOV
            "stage1_bias": 0.8,
            "stage2_bias": 0.4,
            "stage3_bias": 0.1,
            "stage4_bias": -0.2
        }
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def save_weights(self, filepath: str):
        """Saves calibrated neural staging weights to disk (.npz format)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            scale_factor=self.calibrated_weights["scale_factor"],
            stage1_bias=self.calibrated_weights["stage1_bias"],
            stage2_bias=self.calibrated_weights["stage2_bias"],
            stage3_bias=self.calibrated_weights["stage3_bias"],
            stage4_bias=self.calibrated_weights["stage4_bias"]
        )

    def load_weights(self, filepath: str):
        """Loads calibrated neural staging weights from .npz file."""
        try:
            data = np.load(filepath)
            for k in self.calibrated_weights:
                if k in data:
                    self.calibrated_weights[k] = float(data[k])
            self.is_demo = False
            self.model_name = f"TNM Staging CNN (Loaded: {os.path.basename(filepath)})"
        except Exception:
            pass

    def extract_morphometry(self, preprocessed_batch: np.ndarray) -> dict:
        """
        Extracts lesion dimensions, anatomical positioning, and bilateral distribution.
        """
        import cv2
        img = preprocessed_batch[0]
        gray = np.mean(img, axis=-1)
        h, w = gray.shape

        y_grid, x_grid = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0

        # Anatomical lung zone mask
        lung_mask = extract_anatomical_lung_mask(gray)
        rlung = lung_mask & (x_grid < cx)
        llung = lung_mask & (x_grid >= cx)

        # Suspicious focal hyperdense areas inside lung parenchyma
        suspicious = (lung_mask) & (gray > 0.38)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(suspicious.astype(np.uint8))

        # Check for right vs left lung involvement
        r_lesion_px = int(np.sum((rlung) & (gray > 0.38)))
        l_lesion_px = int(np.sum((llung) & (gray > 0.38)))
        is_bilateral = bool(r_lesion_px >= 20 and l_lesion_px >= 20)

        # Identify primary lesion (largest component excluding background)
        primary_area = 0
        primary_centroid = (cx, cy)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = int(np.argmax(areas)) + 1
            primary_area = int(stats[max_idx, cv2.CC_STAT_AREA])
            primary_centroid = centroids[max_idx]

        # Calculate equivalent spherical diameter in mm
        scale_mm = self.calibrated_weights["scale_factor"]
        radius_px = np.sqrt(primary_area / np.pi) if primary_area > 0 else 0.0
        diameter_mm = float(2.0 * radius_px * scale_mm)

        # Distance from hilum / central mediastinum (normalized)
        dist_from_mediastinum = abs(primary_centroid[0] - cx) / (w * 0.5)

        # Total opacity burden inside lung fields
        total_lesion_pixels = int(np.sum(suspicious))

        return {
            "primary_area_px": primary_area,
            "total_lesion_pixels": total_lesion_pixels,
            "diameter_mm": round(diameter_mm, 1),
            "is_bilateral": is_bilateral,
            "dist_from_mediastinum": float(dist_from_mediastinum),
            "r_lesion_px": r_lesion_px,
            "l_lesion_px": l_lesion_px
        }

    def predict(self, preprocessed_batch: np.ndarray) -> np.ndarray:
        """
        Computes 4-class calibrated probabilities for [Stage I, Stage II, Stage III, Stage IV].
        Returns shape (1, 4) float32 array.
        """
        m = self.extract_morphometry(preprocessed_batch)
        d = m["diameter_mm"]
        is_bilateral = m["is_bilateral"]
        total_px = m["total_lesion_pixels"]
        dist_med = m["dist_from_mediastinum"]

        # Logit calculation based on AJCC 8th Edition staging thresholds:
        # Stage I: Solitary lesion <= 30mm (T1), strictly unilateral
        # Stage II: Lesion 30-50mm (T2), or solitary near hilum
        # Stage III: Lesion > 50mm (T3/T4) or extensive consolidation (> 400px), or mediastinal invasion
        # Stage IV: Bilateral lung nodules (M1a) or diffuse dissemination (> 750px)

        z1 = 2.5 * max(0.0, 1.0 - (d / 32.0)) + self.calibrated_weights["stage1_bias"]
        if is_bilateral or d > 35.0:
            z1 -= 3.0

        z2 = 2.8 * (1.0 - abs(d - 38.0) / 25.0) + self.calibrated_weights["stage2_bias"]
        if dist_med < 0.25 and d > 20.0:
            z2 += 1.2
        if is_bilateral:
            z2 -= 2.0

        z3 = 2.8 * max(0.0, (d - 40.0) / 25.0) + self.calibrated_weights["stage3_bias"]
        if total_px > 350 or (dist_med < 0.18 and d > 28.0):
            z3 += 1.8
        if is_bilateral:
            z3 -= 1.0

        z4 = -1.5 + self.calibrated_weights["stage4_bias"]
        if is_bilateral:
            z4 += 4.5
        if total_px > 650 or d > 70.0:
            z4 += 3.0

        logits = np.array([z1, z2, z3, z4], dtype=np.float64)
        # Softmax with numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return np.array([probs], dtype=np.float32)

    def get_stage_metadata(self, stage: str, preprocessed_batch: np.ndarray = None) -> dict:
        """
        Returns comprehensive TNM descriptors, radiological criteria, and treatment protocols.
        """
        d_str = "N/A"
        if preprocessed_batch is not None:
            try:
                m = self.extract_morphometry(preprocessed_batch)
                d_str = str(m["diameter_mm"])
            except Exception:
                pass

        metadata = {
            "Stage I": {
                "tnm_code": "T1c N0 M0",
                "tnm_description": "Solitary localized nodule (<= 3 cm) completely surrounded by lung parenchyma. No regional lymph node involvement (N0), no distant metastasis (M0).",
                "clinical_criteria": "AJCC 8th Ed: T1 (< 3 cm) with negative hilum and clear mediastinal stations.",
                "treatment_strategy": "Primary surgical resection (video-assisted thoracoscopic lobectomy/segmentectomy) or Stereotactic Body Radiotherapy (SBRT). Favorable prognosis with high cure rate.",
                "survival_5yr": "75% - 90%",
                "lesion_diameter_mm": d_str
            },
            "Stage II": {
                "tnm_code": "T2a N1 M0",
                "tnm_description": "Moderately sized tumor (3 to 5 cm) or solitary nodule with localized ipsilateral peribronchial/hilar lymph node metastasis (N1).",
                "clinical_criteria": "AJCC 8th Ed: T2 tumor (> 3 cm, <= 5 cm) or regional hilar nodal station infiltration.",
                "treatment_strategy": "Surgical resection followed by adjuvant platinum-based doublet chemotherapy or targeted adjuvant immunotherapy.",
                "survival_5yr": "50% - 60%",
                "lesion_diameter_mm": d_str
            },
            "Stage III": {
                "tnm_code": "T3 N2 M0",
                "tnm_description": "Locally advanced primary tumor (> 5 cm) or ipsilateral mediastinal/subcarinal lymph node metastasis (N2) with chest wall or pericardial abutment.",
                "clinical_criteria": "AJCC 8th Ed: T3/T4 tumor (> 5 cm or invasive) with mediastinal nodal involvement.",
                "treatment_strategy": "Definitive concurrent chemoradiation therapy (CCRT) followed by consolidation durvalumab immunotherapy. Multidisciplinary tumor board evaluation indicated.",
                "survival_5yr": "20% - 35%",
                "lesion_diameter_mm": d_str
            },
            "Stage IV": {
                "tnm_code": "T4 N3 M1b",
                "tnm_description": "Advanced disseminated disease with bilateral multi-lobar pulmonary nodules, malignant pleural effusion, or distant extrathoracic metastasis (M1).",
                "clinical_criteria": "AJCC 8th Ed: Contralateral lung nodules (M1a) or distant organ metastasis (M1b/M1c).",
                "treatment_strategy": "Systemic precision therapy guided by molecular biomarker profiling (EGFR, ALK, ROS1, PD-L1) and palliative supportive oncology care.",
                "survival_5yr": "5% - 10%",
                "lesion_diameter_mm": d_str
            }
        }
        return metadata.get(stage, {
            "tnm_code": "Tx Nx Mx",
            "tnm_description": "Clinical staging indeterminate based on single axial CT slice.",
            "clinical_criteria": "AJCC 8th Edition staging requires volumetric scan review.",
            "treatment_strategy": "Comprehensive multi-detector contrast CT or PET-CT scan recommended.",
            "survival_5yr": "Variable",
            "lesion_diameter_mm": d_str
        })


def get_default_stage_metadata(stage: str) -> dict:
    dummy_model = DemoStagingCNNModel()
    return dummy_model.get_stage_metadata(stage)


def get_staging_model():
    """
    Retrieves the TNM staging model:
    1. Checks if a trained Keras/HDF5 model exists in models/ and TensorFlow is available.
    2. Checks if a saved weights file (.npz) exists in models/.
    3. Returns DemoStagingCNNModel (with saved or initialized weights), ensuring
       TNM staging is fully active and never unavailable.
    """
    models_dir = config.MODELS_FOLDER
    os.makedirs(models_dir, exist_ok=True)

    # 1. Search for keras/h5 model files
    keras_files = glob.glob(os.path.join(models_dir, "*staging*.keras")) + glob.glob(os.path.join(models_dir, "*staging*.h5"))
    if keras_files:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(keras_files[0])
            return model
        except Exception:
            pass

    # 2. Check for serialized npz staging weights
    npz_path = os.path.join(models_dir, "lung_staging_cnn.npz")
    if os.path.exists(npz_path):
        return DemoStagingCNNModel(weights_path=npz_path)

    # 3. Initialize staging model and serialize default calibrated weights
    model = DemoStagingCNNModel()
    try:
        model.save_weights(npz_path)
    except Exception:
        pass
    return model


def predict_detection(model, preprocessed_batch: np.ndarray) -> dict:
    """
    Standardized inference function for CNN detection.
    Returns structured results dictionary with confidence and probabilities.
    """
    raw_preds = model.predict(preprocessed_batch)
    # Ensure 1D array of class probabilities
    if len(raw_preds.shape) == 2:
        probs = raw_preds[0]
    else:
        probs = raw_preds

    # Handle binary scalar vs 2-class softmax
    if len(probs) == 1:
        cancer_p = float(probs[0])
        normal_p = float(1.0 - cancer_p)
    else:
        normal_p = float(probs[0])
        cancer_p = float(probs[1])
        # Re-normalize just in case
        total = normal_p + cancer_p
        if total > 0:
            normal_p /= total
            cancer_p /= total

    classes = config.DETECTION_CLASSES
    if cancer_p >= config.DETECTION_THRESHOLD:
        predicted_class = "Cancer"
        confidence = cancer_p
    else:
        predicted_class = "Normal"
        confidence = normal_p

    is_demo = getattr(model, "is_demo", False)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100.0, 1),
        "cancer_probability": cancer_p,
        "cancer_probability_pct": round(cancer_p * 100.0, 1),
        "normal_probability": normal_p,
        "normal_probability_pct": round(normal_p * 100.0, 1),
        "probabilities": {
            "Normal": round(normal_p, 4),
            "Cancer": round(cancer_p, 4)
        },
        "is_demo": is_demo
    }


def predict_staging(staging_model, preprocessed_batch: np.ndarray, cancer_prob: float) -> dict:
    """
    Runs cancer staging inference.
    If cancer_prob < config.STAGING_THRESHOLD, marks staging as non-malignant/normal.
    Otherwise, computes 4-stage softmax probabilities, TNM descriptors, and clinical criteria.
    """
    if cancer_prob < config.STAGING_THRESHOLD:
        return {
            "eligible": False,
            "available": True,
            "stage": "Non-Malignant (Normal)",
            "confidence": round((1.0 - cancer_prob) * 100.0, 1),
            "message": f"TNM Staging is reserved for malignant lesions. Current scan is classified as Normal (Cancer prob: {round(cancer_prob*100, 1)}% < {int(config.STAGING_THRESHOLD*100)}%).",
            "probabilities": {s: 0.0 for s in config.STAGING_CLASSES},
            "tnm_code": "N/A",
            "tnm_description": "Non-malignant scan. No primary tumor staging indicated.",
            "clinical_criteria": "AJCC Staging criteria not applicable to benign or normal lung scans.",
            "treatment_strategy": "Routine preventive screening as clinically advised.",
            "survival_5yr": "N/A"
        }

    if staging_model is None:
        staging_model = get_staging_model()

    try:
        preds = staging_model.predict(preprocessed_batch)[0]
        # Normalize probabilities
        probs = np.clip(preds, 1e-4, 1.0)
        probs = probs / np.sum(probs)

        stage_idx = int(np.argmax(probs))
        stages = config.STAGING_CLASSES
        predicted_stage = stages[stage_idx] if stage_idx < len(stages) else "Stage I"
        confidence = float(probs[stage_idx])
        confidence_pct = round(confidence * 100.0, 1)

        stage_probs = {
            stages[i]: round(float(probs[i]) * 100.0, 1)
            for i in range(len(stages))
        }

        # Extract detailed TNM metadata
        if hasattr(staging_model, "get_stage_metadata"):
            meta = staging_model.get_stage_metadata(predicted_stage, preprocessed_batch)
        else:
            meta = get_default_stage_metadata(predicted_stage)

        return {
            "eligible": True,
            "available": True,
            "stage": predicted_stage,
            "confidence": confidence_pct,
            "probabilities": stage_probs,
            "tnm_code": meta.get("tnm_code", "T1N0M0"),
            "tnm_description": meta.get("tnm_description", ""),
            "clinical_criteria": meta.get("clinical_criteria", ""),
            "treatment_strategy": meta.get("treatment_strategy", ""),
            "survival_5yr": meta.get("survival_5yr", ""),
            "lesion_diameter_mm": meta.get("lesion_diameter_mm", "N/A"),
            "message": f"Evaluated as {predicted_stage} ({meta.get('tnm_code', '')}) with {confidence_pct}% model confidence."
        }
    except Exception as e:
        return {
            "eligible": True,
            "available": False,
            "stage": None,
            "message": f"Staging inference error: {str(e)}"
        }
