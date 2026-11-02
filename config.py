import os

# Base directory of the application
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Security and session
SECRET_KEY = os.environ.get("SECRET_KEY", "lung-cancer-xai-counterfactual-secret-key-2026")

# Directory configurations
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GENERATED_FOLDER = os.path.join(BASE_DIR, "generated")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
CASES_FOLDER = os.path.join(STATIC_FOLDER, "images", "cases")
PROGRESSION_FOLDER = os.path.join(STATIC_FOLDER, "images", "progression")

# File upload restrictions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "dcm"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Model configurations
IMAGE_SIZE = (224, 224)  # Height, Width expected by CNN
CHANNELS = 3  # RGB channels
DETECTION_CLASSES = ["Normal", "Cancer"]
DETECTION_MODEL_FILENAME = "lung_cancer_cnn.keras"
STAGING_MODEL_FILENAME = "lung_staging_cnn.keras"
DETECTION_THRESHOLD = 0.50

# Staging configuration
STAGING_THRESHOLD = 0.50  # If cancer probability >= 50%, trigger staging attempt
STAGING_CLASSES = ["Stage I", "Stage II", "Stage III", "Stage IV"]

# Counterfactual What-If configurations
COUNTERFACTUAL_MODES = [
    "Mask suspicious region",
    "Blur suspicious region",
    "Reduce intensity in suspicious region"
]

COUNTERFACTUAL_STRENGTHS = {
    "Low": 0.35,
    "Medium": 0.65,
    "High": 0.95
}

DEFAULT_MODE = "Mask suspicious region"
DEFAULT_STRENGTH = "Medium"

# Similar Case Retrieval Configurations
SIMILAR_CASES_TOP_K = 4

# Disease Progression Simulator Configurations
PROGRESSION_TIMEPOINTS_MIN = 2
PROGRESSION_TIMEPOINTS_MAX = 5
RECIST_PROGRESSIVE_THRESHOLD = 20.0  # +20% diameter increase
RECIST_PARTIAL_RESPONSE_THRESHOLD = -30.0  # -30% diameter decrease

# Academic & Safety Metadata
PROJECT_TITLE = "Lung Cancer Detection and Staging Using CNN with Counterfactual What-If Explainable Analysis"
PROJECT_SHORT_TITLE = "Lung Cancer XAI & Comparative Oncology"
ACADEMIC_YEAR = "Final Year Academic Capstone Project"
DISCLAIMER = (
    "This application is an academic research prototype and does not provide a real medical diagnosis. "
    "Results are for educational and research explainability demonstration purposes only. "
    "Always consult a qualified radiologist or medical professional for clinical diagnosis."
)
DEMO_MODE_NOTICE = "DEMO MODE - NOT A REAL MEDICAL DIAGNOSIS"
