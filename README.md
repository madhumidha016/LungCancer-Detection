# Lung Cancer Detection and Staging Using CNN with Counterfactual What-If Explainable Analysis

An end-to-end academic research prototype combining Convolutional Neural Networks (CNNs), Grad-CAM visual attention mapping, and **transparent counterfactual lesion perturbation** to verify causal diagnostic decisions on chest Computed Tomography (CT) scans.

---

## 🌟 Key Features & Innovations

1. **Dual Ingestion Engine:**
   - Standard image upload supporting PNG, JPG, JPEG, WEBP, BMP, TIFF, and DICOM (DCM) slices with automatic format and dimension validation.
   - Built-in calibrated benchmark samples (Normal clear lung vs Suspicious malignant nodule) for instant one-click testing.

2. **CNN-Based Detection Pipeline:**
   - Standardized $224 \times 224 \times 3$ RGB preprocessing and normalized tensor preparation.
   - Dual-class calibrated softmax inference (`Normal` vs `Cancer`) with real-time confidence metrics.

3. **Grad-CAM Visual Explainability:**
   - Gradient-weighted Class Activation Mapping (Grad-CAM) localizing the receptive fields that contributed most strongly to the malignancy prediction.
   - Visual tri-gallery presenting the Preprocessed CT, raw Jet Colormap Heatmap, and anatomical Diagnostic Overlay.

4. **Counterfactual What-If Causal Analysis (The Core Novelty):**
   - Moves beyond correlational heatmaps to verify **causality**.
   - Applies transparent in-silico interventions to the suspicious lesion:
     - **Mask suspicious region:** Blends lesion area with healthy aerated lung parenchyma.
     - **Blur suspicious region:** Dissolves irregular spiculation margins and high-frequency textures.
     - **Reduce intensity in suspicious region:** Simulates soft-tissue radiodensity attenuation and lesion regression.
   - Re-evaluates the modified CT slice through the **identical CNN detector**.
   - Determines whether the prediction shifted or flipped (`Cancer` &rarr; `Normal`), with automated clinical reasoning.

5. **Analytical Visualization & Reporting:**
   - Automated Matplotlib side-by-side probability shift comparison chart.
   - Optional TNM Staging assessment (Stages I-IV).
   - Instant downloadable formal academic text report and printable clinical summary.

6. **Interactive Academic AI Chatbot:**
   - Embedded floating assistant answering queries on CNNs, Grad-CAM equations, What-If perturbation methodology, staging criteria, and medical safety.

---

## 🚀 Quick Start & Environment Setup (Windows)

Follow these exact Windows commands in PowerShell or Command Prompt:

```powershell
# 1. Verify Python installation
python --version

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install project dependencies
pip install -r requirements.txt

# 6. Launch the Flask application
python app.py
```

### Accessing the Web Application
Once the server starts, open your web browser (Google Chrome recommended) and navigate to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🧪 Running Automated Tests

To execute the automated test suite verifying all routes, uploads, inference, Grad-CAM, What-If analysis, report downloads, and chatbot APIs:

```powershell
venv\Scripts\pytest tests/test_app.py -v
```

---

## 🧠 Model Integration Guide

- **Safe Demo Mode:** If no model weights are found in `models/`, the application runs in safe **DEMO MODE**. The UI clearly displays `DEMO MODE - NOT A REAL MEDICAL DIAGNOSIS` while maintaining a fully functional feature extraction simulation.
- **Trained Model Mode:** To connect your trained Keras/TensorFlow model, place your file as `lung_cancer_cnn.keras` or `lung_cancer_cnn.h5` inside the `models/` directory. The application will detect and load it automatically upon launch.

---

## 📂 Project Architecture

```
lung/
├── app.py                  # Main Flask application and routing
├── config.py               # Centralized configuration and thresholds
├── preprocessing.py        # Image validation, resizing, and normalization
├── model_utils.py          # CNN detection, demo simulation, and staging
├── gradcam.py              # Grad-CAM heatmap and overlay generation
├── counterfactual.py       # Counterfactual What-If perturbation engine
├── chart_utils.py          # Matplotlib probability comparison charts
├── report_generator.py     # Academic report compilation and export
├── chatbot.py              # Academic knowledge base and FAQ assistant
├── requirements.txt        # Production Python dependencies
├── README.md               # Setup and architecture documentation
├── .gitignore              # Git ignore rules
│
├── templates/              # Jinja2 HTML templates
│   ├── base.html           # Master layout with navbar, chatbot, and footer
│   ├── index.html          # Project home and hero overview
│   ├── about.html          # Detailed academic and theoretical methodology
│   ├── predict.html        # CT scan upload and benchmark sample selector
│   ├── result.html         # Detection metrics and Grad-CAM tri-gallery
│   ├── counterfactual.html # Side-by-side What-If comparison and chart
│   └── report.html         # Comprehensive printable academic report
│
├── static/
│   ├── css/
│   │   └── style.css       # Complete modern dark slate design system
│   ├── js/
│   │   └── app.js          # Client-side preview, upload, and chatbot logic
│   └── images/             # Sample benchmark CT scans
│
├── models/
│   └── README.md           # Model weight placement instructions
│
├── uploads/                # User uploaded CT scans
├── generated/              # Preprocessed views, Grad-CAM overlays, What-If outputs
├── reports/                # Exported text and PDF academic reports
└── tests/
    └── test_app.py         # Automated test suite
```

---

## ⚖️ Medical Safety Disclaimer

> **IMPORTANT:** This software application is an academic capstone research prototype developed solely for educational, scientific evaluation, and Explainable AI demonstration purposes. It does **not** provide a real medical diagnosis, is not clinical software, and must not be used for patient care or medical decision-making. Always consult a qualified radiologist or medical specialist for clinical interpretation.
