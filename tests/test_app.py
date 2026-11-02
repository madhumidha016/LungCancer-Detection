import io
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from PIL import Image
import numpy as np
from app import app
import config

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        with app.app_context():
            yield client

def create_test_image_bytes(color=(128, 128, 128), size=(224, 224), format="PNG"):
    """Helper to generate in-memory image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf

# 1. Page Load Tests
def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Lung Cancer Detection" in response.data
    assert b"Counterfactual What-If" in response.data

def test_about_page_redirects(client):
    response = client.get("/about", follow_redirects=True)
    assert response.status_code == 200
    assert b"Lung Cancer Detection" in response.data

def test_predict_page_get(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Upload CT Image File" in response.data

# 2. Upload Validation Tests
def test_upload_invalid_extension(client):
    data = {
        "file": (io.BytesIO(b"dummy text file"), "test.txt")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"File format not allowed" in response.data

def test_upload_empty_submission(client):
    response = client.post("/predict", data={}, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"No file selected" in response.data or b"No file part" in response.data

# 3. Sample CT Prediction & Full Pipeline Test
def test_sample_cancer_prediction(client):
    response = client.post("/predict", data={"sample_choice": "sample_cancer"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Diagnostic Inference & Grad-CAM" in response.data
    assert b"Cancer" in response.data
    assert b"Grad-CAM Heatmap" in response.data

def test_sample_normal_prediction(client):
    response = client.post("/predict", data={"sample_choice": "sample_normal"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Diagnostic Inference & Grad-CAM" in response.data
    assert b"Normal" in response.data

# 4. Counterfactual What-If Execution Test
def test_counterfactual_whatif_pipeline(client):
    # First, run a sample prediction to establish session
    client.post("/predict", data={"sample_choice": "sample_cancer"}, follow_redirects=True)

    # Check GET /counterfactual
    cf_get = client.get("/counterfactual")
    assert cf_get.status_code == 200
    assert b"Counterfactual What-If" in cf_get.data

    # POST to execute What-If
    cf_post = client.post("/counterfactual", data={
        "mode": "Mask suspicious region",
        "strength": "High"
    }, follow_redirects=True)

    assert cf_post.status_code == 200
    assert b"Prediction Changed:" in cf_post.data
    assert b"Original vs Counterfactual CT" in cf_post.data
    assert b"Probability Distribution Shift" in cf_post.data

# 5. Report Compilation and Download Test
def test_report_generation_and_download(client):
    # Establish session with prediction and counterfactual
    client.post("/predict", data={"sample_choice": "sample_cancer"}, follow_redirects=True)
    client.post("/counterfactual", data={"mode": "Mask suspicious region", "strength": "Medium"}, follow_redirects=True)

    # View report page
    rep_res = client.get("/report")
    assert rep_res.status_code == 200
    assert b"EVALUATION REPORT" in rep_res.data.upper()
    assert b"Report ID:" in rep_res.data

    # Check that download route works
    report_files = os.listdir(config.REPORTS_FOLDER)
    assert len(report_files) > 0
    test_report_file = report_files[0]

    dl_res = client.get(f"/report/download/{test_report_file}")
    assert dl_res.status_code == 200
    assert b"SECTION 1: CNN DETECTION INFERENCE" in dl_res.data

# 6. Chatbot API Tests
def test_chatbot_api_query(client):
    res = client.post("/api/chatbot", json={"message": "What is counterfactual analysis?"})
    assert res.status_code == 200
    data = res.get_json()
    assert "Counterfactual" in data["topic"]
    assert "causality" in data["answer"].lower() or "intervention" in data["answer"].lower()

def test_chatbot_api_suggestions(client):
    res = client.get("/api/chatbot/suggestions")
    assert res.status_code == 200
    data = res.get_json()
    assert "suggestions" in data
    assert len(data["suggestions"]) > 0

# 7. Safe Fallbacks & Staging Model Tests
def test_missing_model_safety(client):
    from model_utils import get_detection_model
    model, is_demo, meta = get_detection_model()
    # In demo mode, it should be marked as demo without throwing
    assert is_demo is True
    assert "DEMO" in meta["status"]

def test_staging_model_inference():
    from model_utils import get_staging_model, predict_staging
    from preprocessing import load_and_preprocess_image
    
    staging_model = get_staging_model()
    assert staging_model is not None

    batch, _ = load_and_preprocess_image("static/images/sample_cancer.png")
    # For cancer sample with high cancer probability
    result = predict_staging(staging_model, batch, cancer_prob=0.95)
    assert result["eligible"] is True
    assert result["available"] is True
    assert result["stage"] in config.STAGING_CLASSES
    assert result["confidence"] > 0
    assert "tnm_code" in result
    assert "Stage" in result["message"]

def test_sample_cancer_displays_staging_in_ui(client):
    res = client.post("/predict", data={"sample_choice": "sample_cancer"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"TNM Staging Assessment" in res.data
    assert b"CNN Staging Active" in res.data
    assert b"Stage" in res.data
    assert b"Pathological Criteria" in res.data
