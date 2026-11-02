import re
import config

KNOWLEDGE_BASE = [
    {
        "intent": "project_overview",
        "keywords": ["how does this project work", "about project", "overview", "workflow", "how it works", "what does this app do"],
        "topic": "Project Architecture & Workflow",
        "answer": (
            "This project implements an end-to-end academic framework for Lung Cancer Detection and Staging "
            "using Convolutional Neural Networks (CNNs) coupled with Counterfactual What-If Explainability. "
            "The workflow consists of: (1) Image Validation & Preprocessing of chest CT scans, (2) CNN Classification "
            "into Normal vs Cancer with confidence scoring, (3) Grad-CAM localization highlighting suspicious opacities, "
            "(4) Counterfactual What-If Intervention where suspicious regions are perturbed (masked/blurred/attenuated) "
            "and re-evaluated through the same model to test causal reasoning, (5) Optional Staging evaluation, "
            "and (6) Automated Academic Report Generation."
        ),
        "follow_ups": ["What is counterfactual analysis?", "What is Grad-CAM?", "How do I load a trained model?"]
    },
    {
        "intent": "counterfactual_whatif",
        "keywords": ["counterfactual", "what if", "what-if", "novelty", "perturbation", "mask suspicious", "blur suspicious"],
        "topic": "Counterfactual What-If Analysis (Core Novelty)",
        "answer": (
            "Counterfactual What-If Analysis is the primary novelty of this research. While traditional heatmaps (like Grad-CAM) "
            "only indicate correlation, counterfactual analysis tests CAUSALITY. By systematically masking, blurring, or attenuating "
            "the suspicious lesion identified by Grad-CAM and running the modified image back through the identical CNN detector, "
            "we observe whether the model's cancer probability drops and if the classification flips from Cancer to Normal. "
            "This provides verifiable evidence that the identified nodule was truly the causal determinant of the prediction."
        ),
        "follow_ups": ["How does Grad-CAM work?", "What counterfactual modes exist?", "What is CNN?"]
    },
    {
        "intent": "gradcam_explainability",
        "keywords": ["grad-cam", "gradcam", "heatmap", "overlay", "activation map", "explainability", "xai"],
        "topic": "Grad-CAM Explainability",
        "answer": (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) is an Explainable AI (XAI) technique for deep neural networks. "
            "It computes the gradients of the target class score with respect to the feature maps of the final convolutional layer. "
            "These gradients act as importance weights, producing a coarse 2D heatmap that visualizes the discriminative regions "
            "in the CT slice (such as solitary pulmonary nodules or ground-glass opacities) that led the network to its decision."
        ),
        "follow_ups": ["What is counterfactual analysis?", "What is CNN?", "What is lung cancer detection?"]
    },
    {
        "intent": "cnn_detection",
        "keywords": ["what is cnn", "convolutional neural network", "deep learning", "architecture", "layers", "model"],
        "topic": "Convolutional Neural Networks (CNNs)",
        "answer": (
            "A Convolutional Neural Network (CNN) is a specialized deep learning architecture designed for image processing. "
            "It utilizes convolutional filters to automatically and adaptively learn spatial hierarchies of features from chest CT scans. "
            "Low-level layers capture edges and texture gradients, intermediate layers detect pulmonary parenchyma patterns, "
            "and deeper layers recognize complex pathological structures like spiculated malignant masses."
        ),
        "follow_ups": ["How does this project work?", "What is lung cancer detection?", "What is staging?"]
    },
    {
        "intent": "lung_cancer_detection",
        "keywords": ["lung cancer", "ct scan", "detection", "nodule", "malignancy", "computed tomography"],
        "topic": "Lung Cancer Detection & CT Imaging",
        "answer": (
            "Lung cancer detection on Computed Tomography (CT) involves screening axial chest slices for pulmonary nodules, "
            "subsolid opacities, and architectural distortions. Early detection significantly improves 5-year survival rates. "
            "Computer-Aided Detection (CAD) systems powered by CNNs assist clinicians by providing rapid second-look detection "
            "and quantitative probability metrics."
        ),
        "follow_ups": ["What is staging?", "What is counterfactual analysis?", "Is this a medical diagnosis?"]
    },
    {
        "intent": "cancer_staging",
        "keywords": ["staging", "stage i", "stage ii", "stage iii", "stage iv", "tnm", "how does staging work"],
        "topic": "Lung Cancer Staging",
        "answer": (
            "Cancer staging categorizes the extent and anatomical spread of malignancy based on the AJCC/UICC 8th Edition TNM system "
            "(Tumor size, Node involvement, and Metastasis). Stages range from Stage I (localized solitary nodule ≤3 cm) to Stage IV "
            "(bilateral or metastatic spread). In this application, an active TNM Staging CNN Morphometry Engine automatically "
            "evaluates lesion dimensions, focal radiodensity, and bilateral lung distribution when Cancer is detected (≥50% probability). "
            "It computes multi-class stage probabilities across Stages I-IV and provides AJCC TNM clinical criteria."
        ),
        "follow_ups": ["What is counterfactual analysis?", "What is Grad-CAM?", "Is this a medical diagnosis?"]
    },
    {
        "intent": "medical_safety",
        "keywords": ["medical diagnosis", "is this real", "doctor", "clinical use", "disclaimer", "fda", "hospital"],
        "topic": "Medical Safety & Prototype Scope",
        "answer": (
            f"DISCLAIMER: {config.DISCLAIMER}\n\n"
            "This software is an academic capstone research prototype engineered to demonstrate Explainable AI and counterfactual reasoning. "
            "It is NOT a medical device and must never be used for clinical decision-making or self-diagnosis."
        ),
        "follow_ups": ["How does this project work?", "What is counterfactual analysis?"]
    },
    {
        "intent": "model_replacement",
        "keywords": ["load model", "trained model", "replace model", "weights", ".keras", ".h5", "models folder"],
        "topic": "Model Integration & Architecture",
        "answer": (
            "To connect a real trained CNN model, save your weights file as 'lung_cancer_cnn.keras' or 'lung_cancer_cnn.h5' "
            "directly inside the 'models/' directory. The system automatically inspects the directory upon inference. "
            "If no weights are provided, the system executes in safe DEMO MODE so all workflows, heatmaps, and counterfactual transformations "
            "remain fully testable and responsive."
        ),
        "follow_ups": ["How does this project work?", "What is counterfactual analysis?"]
    }
]

DEFAULT_RESPONSE = {
    "topic": "Academic Project Support",
    "answer": (
        "I am the Academic Assistant for the 'Lung Cancer Detection with Counterfactual What-If Analysis' project. "
        "I can help explain our CNN detection pipeline, Grad-CAM visualization, Counterfactual What-If perturbation logic, "
        "cancer staging, or medical safety considerations. Try asking one of the suggested questions below!"
    ),
    "follow_ups": [
        "What is counterfactual analysis?",
        "What is Grad-CAM?",
        "How does this project work?",
        "What is staging?",
        "Is this a medical diagnosis?"
    ]
}


def get_chatbot_response(user_query: str) -> dict:
    """
    Evaluates user query against project knowledge base using normalized regex and token overlap.
    Returns structured answer payload.
    """
    if not user_query or not user_query.strip():
        return DEFAULT_RESPONSE

    cleaned_query = re.sub(r"[^\w\s]", "", user_query.lower()).strip()
    query_tokens = set(cleaned_query.split())

    best_match = None
    best_score = 0

    for item in KNOWLEDGE_BASE:
        score = 0
        for kw in item["keywords"]:
            kw_clean = re.sub(r"[^\w\s]", "", kw.lower())
            # Exact phrase match gives highest weight
            if kw_clean in cleaned_query:
                score += 8
            # Token overlap gives secondary weight
            kw_tokens = set(kw_clean.split())
            overlap = len(query_tokens.intersection(kw_tokens))
            score += overlap * 2

        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= 2:
        return {
            "topic": best_match["topic"],
            "answer": best_match["answer"],
            "follow_ups": best_match["follow_ups"]
        }

    return DEFAULT_RESPONSE


def get_suggested_queries() -> list:
    """Returns list of common questions for quick-access chips."""
    return [
        "What is counterfactual analysis?",
        "How does this project work?",
        "What is Grad-CAM?",
        "What is CNN?",
        "What is staging?",
        "Is this a medical diagnosis?"
    ]
