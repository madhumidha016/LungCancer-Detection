# Models Directory - Trained Model Integration Guide

This directory is monitored by the LUNG framework for real trained deep learning model files.

---

## 1. Operating Modes

### A. Safe Academic DEMO MODE (Default when this folder is empty)
- When no model weights are found in this directory, the application automatically runs in **DEMO MODE**.
- The user interface clearly displays:
  `DEMO MODE - NOT A REAL MEDICAL DIAGNOSIS`
- The pipeline executes a simulated density-convolution feature extraction engine so that all CT preprocessing, Grad-CAM visual attention mapping, Counterfactual What-If in-silico perturbations, comparison charts, and reports remain fully testable and responsive.

### B. Live Trained Model Mode
- When a compatible trained model file is placed in this directory, the system automatically detects and loads it upon restart/inference.
- The UI switches from Demo Mode to `TRAINED CNN ACTIVE`.

---

## 2. Supported File Formats

The detection model loader supports:
1. **Keras v3 Format (Recommended):** `lung_cancer_cnn.keras`
2. **HDF5 Format:** `lung_cancer_cnn.h5`
3. **TensorFlow SavedModel:** Directory containing `saved_model.pb` and `variables/`

---

## 3. Expected Model Architecture Specifications

### Detection Model
- **File Name:** `lung_cancer_cnn.keras` or `lung_cancer_cnn.h5`
- **Input Shape:** `(None, 224, 224, 3)` (Standard 3-channel RGB CT slice normalized between `0.0` and `1.0`)
- **Output Layer:**
  - Option 1 (2-class Softmax): Shape `(None, 2)` representing `[Normal, Cancer]`
  - Option 2 (Sigmoid Binary): Shape `(None, 1)` representing Cancer probability

### Optional Staging Model
- **File Name:** `lung_staging_cnn.keras` or `lung_staging_cnn.h5`
- **Input Shape:** `(None, 224, 224, 3)`
- **Output Layer:** Shape `(None, 4)` representing `[Stage I, Stage II, Stage III, Stage IV]`

---

## 4. How to Export from Keras / TensorFlow

```python
# In your training script / notebook:
model.save("models/lung_cancer_cnn.keras")
```

Once saved, restart the Flask web application:
```bash
python app.py
```
The application will automatically detect and load your trained weights without modifying any application code.
