# Lung Cancer Detection and Staging Using CNN with Counterfactual What-If, Similar Case Retrieval & Progression Simulator

An end-to-end clinical research platform combining Convolutional Neural Networks (CNNs), Grad-CAM visual attention mapping, **transparent counterfactual lesion perturbation**, **Content-Based Similar Case Retrieval (CBIR)**, and **Longitudinal Disease Progression Simulation** on chest Computed Tomography (CT) scans.

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

5. **Similar Case Retrieval System (Content-Based Image Retrieval - CBIR):**
   - When a scan is uploaded, the system computes a 5-dimensional radiomic embedding (malignancy probability, nodule diameter, attenuation density, margin spiculation, and TNM stage).
   - Searches a multi-centric benchmark reference cohort to retrieve radiomically concordant historical cases.
   - Displays confirmed histopathological diagnoses (Acinar Adenocarcinoma, Squamous Cell Carcinoma, Benign Hamartoma, Granuloma, etc.).
   - Presents real-world treatment regimens (VATS Lobectomy, SBRT, Adjuvant Osimertinib, Pembrolizumab immunotherapy) and clinical outcomes (5-year disease-free survival, RECIST response, tolerability) to give physicians comparative insights.

6. **Longitudinal Disease Progression Simulator:**
   - Ingests serial CT scans taken across clinical follow-up intervals (e.g. Baseline, Month 3, Month 6).
   - Quantifies tumor growth metrics: longest unidimensional diameter (mm), cross-sectional area ($mm^2$), and ellipsoidal 3D volume ($cm^3$).
   - Calculates Volume Doubling Time (VDT) via Schwartz's kinetic formula and classifies response under standard **RECIST 1.1** criteria (Progressive Disease $\ge +20\%$, Stable Disease, Partial Response $\le -30\%$).
   - Generates multi-panel Matplotlib progression charts with historical measurements and shaded 95% forecast cones.
   - Employs an exponential/Gompertzian kinetic growth model to predict the next AJCC stage transition horizon (e.g., crossing 30mm for Stage II or 50mm for Stage III) and provides evidence-based clinical action directives.

7. **Analytical Visualization & Reporting:**
   - Automated Matplotlib side-by-side probability shift comparison charts and progression trajectory figures.
   - Optional TNM Staging assessment (Stages I-IV) based on AJCC/UICC 8th Edition criteria.
   - Instant downloadable formal academic text reports and printable clinical summaries.

8. **Interactive Academic AI Chatbot:**
   - Embedded floating assistant answering queries on CNNs, Grad-CAM equations, What-If perturbation methodology, Similar Case Retrieval, progression kinetics, staging criteria, and medical safety.

---

## 🚀 Quick Start & Environment Setup (Windows)

Follow these exact Windows commands in PowerShell or Command Prompt:

```powershell
# 1. Verify Python installation (Python 3.12 recommended)
python --version

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\activate

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Launch the Flask application
python app.py
```

### Accessing the Web Application
Once the server starts, open your web browser and navigate to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🧪 Running Automated Tests

To execute the automated test suite verifying all 27 unit and integration tests (CNN detection, Grad-CAM, What-If analysis, Similar Case Retrieval, Disease Progression kinetics, report generation, and chatbot APIs):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 🧠 Model Integration Guide

- **Safe Demo Mode:** If no model weights are found in `models/`, the application runs in safe **DEMO MODE**. The UI clearly displays `DEMO MODE - NOT A REAL MEDICAL DIAGNOSIS` while maintaining fully functional feature extraction and realistic lung morphology simulation.
- **Trained Model Mode:** To connect your trained Keras/TensorFlow model, place your file as `lung_cancer_cnn.keras` or `lung_cancer_cnn.h5` inside the `models/` directory. The application will detect and load it automatically upon launch.

---

## 📂 Project Architecture

```
LungCancer-Detection/
├── app.py                      # Main Flask application and REST routes
├── config.py                   # Centralized configuration and clinical thresholds
├── case_retrieval.py           # Similar Case Retrieval (CBIR) & cohort database
├── progression_simulator.py    # Longitudinal tumor kinetics & next-stage forecasting
├── preprocessing.py            # Image validation, resizing, lung segmentation
├── model_utils.py              # CNN detection, demo simulation, and TNM staging
├── gradcam.py                  # Grad-CAM heatmap and overlay generation
├── counterfactual.py           # Counterfactual What-If perturbation engine
├── chart_utils.py              # Matplotlib probability comparison charts
├── report_generator.py         # Academic report compilation and text export
├── chatbot.py                  # Academic knowledge base and FAQ assistant
├── requirements.txt            # Production Python dependencies
├── README.md                   # Setup and architecture documentation
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Master layout with navbar, chatbot, and footer
│   ├── index.html              # Project hero overview and feature showcase
│   ├── predict.html            # CT scan upload and benchmark sample selector
│   ├── result.html             # Detection metrics, Grad-CAM tri-gallery & similar cases
│   ├── similar_cases.html      # Dedicated Similar Case Retrieval cohort browser
│   ├── progression.html        # Longitudinal Disease Progression Simulator
│   ├── counterfactual.html     # Side-by-side What-If comparison and chart
│   └── report.html             # Comprehensive printable academic report
│
├── static/
│   ├── css/
│   │   └── style.css           # Complete modern dark slate design system
│   ├── js/
│   │   └── app.js              # Client-side preview, upload, and chatbot logic
│   └── images/
│       ├── sample_cancer.png   # Calibrated benchmark malignant slice
│       ├── sample_normal.png   # Calibrated benchmark clear lung slice
│       ├── cases/              # Generated reference case CT thumbnails
│       └── progression/        # Longitudinal progression demo sequential slices
│
├── models/
│   └── README.md               # Model weight placement instructions
│
├── uploads/                    # User uploaded CT scans
├── generated/                  # Preprocessed views, Grad-CAM overlays, progression charts
├── reports/                    # Exported text and clinical academic reports
└── tests/
    ├── test_app.py             # Core application test suite
    └── test_new_features.py    # Similar cases & progression simulator test suite
```

---

## ⚖️ Medical Safety Disclaimer

> **IMPORTANT:** This software application is an academic capstone research prototype developed solely for educational, scientific evaluation, and Explainable AI demonstration purposes. It does **not** provide a real medical diagnosis, is not clinical software, and must not be used for patient care or medical decision-making. Always consult a qualified radiologist or medical specialist for clinical interpretation.
