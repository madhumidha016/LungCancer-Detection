import os
import shutil
from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, session, jsonify, send_from_directory
)
import config
from preprocessing import validate_and_save_upload, load_and_preprocess_image, prepare_display_image
from model_utils import get_detection_model, predict_detection, get_staging_model, predict_staging
from gradcam import generate_gradcam_visuals
from counterfactual import run_whatif_analysis
from chart_utils import generate_comparison_chart
from report_generator import build_report_payload, save_downloadable_report
from chatbot import get_chatbot_response, get_suggested_queries

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Ensure all workspace directories exist
for folder in [config.UPLOAD_FOLDER, config.GENERATED_FOLDER, config.REPORTS_FOLDER, config.MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Context processor for global template variables
@app.context_processor
def inject_global_vars():
    # Detect current model status
    _, is_demo, model_meta = get_detection_model()
    return {
        "PROJECT_TITLE": config.PROJECT_TITLE,
        "PROJECT_SHORT_TITLE": config.PROJECT_SHORT_TITLE,
        "DISCLAIMER": config.DISCLAIMER,
        "DEMO_MODE_NOTICE": config.DEMO_MODE_NOTICE,
        "IS_DEMO_MODE": is_demo,
        "MODEL_META": model_meta
    }


# ==========================================
# STATIC ARTIFACT SERVING ROUTES
# ==========================================
@app.route("/generated/<path:filename>")
def serve_generated(filename):
    return send_from_directory(config.GENERATED_FOLDER, filename)


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(config.UPLOAD_FOLDER, filename)


# ==========================================
# APPLICATION WEB ROUTES
# ==========================================
@app.route("/")
def index():
    """Home page displaying project overview, features, and clinical workflow."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Redirect removed about/methodology page to home."""
    return redirect(url_for("index"))


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Image upload and prediction route."""
    if request.method == "GET":
        return render_template("predict.html")

    # POST request: handle file upload or sample image selection
    try:
        sample_choice = request.form.get("sample_choice")
        if sample_choice in ["sample_normal", "sample_cancer"]:
            sample_source = os.path.join(config.STATIC_FOLDER, "images", f"{sample_choice}.png")
            if not os.path.exists(sample_source):
                flash("Sample image not found on server.", "danger")
                return redirect(url_for("predict_page"))
            
            # Copy sample into uploads
            dest_filename = f"sample_{sample_choice}.png"
            saved_path = os.path.join(config.UPLOAD_FOLDER, dest_filename)
            shutil.copyfile(sample_source, saved_path)
            orig_filename = dest_filename
        else:
            if "file" not in request.files:
                flash("No file part in the request.", "warning")
                return redirect(url_for("predict_page"))

            file = request.files["file"]
            saved_path, orig_filename = validate_and_save_upload(file)

        # Preprocess image for CNN
        preprocessed_batch, display_rgb = load_and_preprocess_image(saved_path)
        _, display_filename = prepare_display_image(saved_path)

        # Run CNN Detection
        model, is_demo, model_info = get_detection_model()
        detection_res = predict_detection(model, preprocessed_batch)

        # Run Grad-CAM visual attention
        target_class_idx = 1 if detection_res["predicted_class"] == "Cancer" else 0
        gradcam_res = generate_gradcam_visuals(
            model=model,
            preprocessed_batch=preprocessed_batch,
            display_image_rgb=display_rgb,
            target_class_index=target_class_idx
        )

        # Optional Staging check
        staging_model = get_staging_model()
        staging_res = predict_staging(
            staging_model=staging_model,
            preprocessed_batch=preprocessed_batch,
            cancer_prob=detection_res["cancer_probability"]
        )

        # Cache session data
        session["current_analysis"] = {
            "image_path": saved_path,
            "orig_filename": orig_filename,
            "display_filename": display_filename,
            "detection_res": detection_res,
            "gradcam_res": {
                "success": gradcam_res["success"],
                "heatmap_filename": gradcam_res.get("heatmap_filename"),
                "overlay_filename": gradcam_res.get("overlay_filename"),
                "message": gradcam_res.get("message")
            },
            "staging_res": staging_res,
            "cf_res": None,
            "chart_filename": None
        }

        # Keep heatmap array in memory or temporary scratch for counterfactual
        if gradcam_res.get("heatmap_array") is not None:
            heat_temp = os.path.join(config.GENERATED_FOLDER, f"temp_heat_{orig_filename}.npy")
            import numpy as np
            np.save(heat_temp, gradcam_res["heatmap_array"])
            session["heatmap_temp_path"] = heat_temp

        flash(f"Inference complete: Predicted {detection_res['predicted_class']} ({detection_res['confidence_pct']}%)", "success")
        return redirect(url_for("result_page"))

    except ValueError as ve:
        flash(str(ve), "danger")
        return redirect(url_for("predict_page"))
    except Exception as e:
        flash(f"An error occurred during image processing: {str(e)}", "danger")
        return redirect(url_for("predict_page"))


@app.route("/result")
def result_page():
    """Displays detection metrics, Grad-CAM heatmap, and overlay."""
    analysis = session.get("current_analysis")
    if not analysis:
        flash("No active analysis found. Please upload a CT scan image first.", "info")
        return redirect(url_for("predict_page"))

    return render_template(
        "result.html",
        analysis=analysis,
        detection=analysis["detection_res"],
        gradcam=analysis["gradcam_res"],
        staging=analysis["staging_res"]
    )


@app.route("/counterfactual", methods=["GET", "POST"])
def counterfactual_page():
    """Counterfactual What-If interface for causal perturbation analysis."""
    analysis = session.get("current_analysis")
    if not analysis:
        flash("Please upload and analyze a CT scan image first.", "info")
        return redirect(url_for("predict_page"))

    if request.method == "POST":
        mode = request.form.get("mode", config.DEFAULT_MODE)
        strength_label = request.form.get("strength", config.DEFAULT_STRENGTH)

        try:
            model, _, _ = get_detection_model()
            _, display_rgb = load_and_preprocess_image(analysis["image_path"])

            # Load cached Grad-CAM heatmap if available
            heatmap_array = None
            heat_path = session.get("heatmap_temp_path")
            if heat_path and os.path.exists(heat_path):
                import numpy as np
                heatmap_array = np.load(heat_path)

            # Run Counterfactual What-If analysis
            cf_res = run_whatif_analysis(
                detection_model=model,
                original_image_path=analysis["image_path"],
                display_image_rgb=display_rgb,
                heatmap_array=heatmap_array,
                mode=mode,
                strength_label=strength_label
            )

            # Generate comparison chart
            _, chart_filename = generate_comparison_chart(
                original_probs=cf_res["original_probabilities"],
                counterfactual_probs=cf_res["counterfactual_probabilities"],
                prediction_changed=cf_res["prediction_changed"]
            )

            # Update session
            analysis["cf_res"] = cf_res
            analysis["chart_filename"] = chart_filename
            session["current_analysis"] = analysis

            flash(f"What-If Analysis Completed: Prediction Changed: {cf_res['prediction_changed_str']}", "success")
            return redirect(url_for("counterfactual_page"))

        except Exception as e:
            flash(f"Error executing counterfactual analysis: {str(e)}", "danger")
            return redirect(url_for("counterfactual_page"))

    return render_template(
        "counterfactual.html",
        analysis=analysis,
        detection=analysis["detection_res"],
        cf_res=analysis.get("cf_res"),
        chart_filename=analysis.get("chart_filename"),
        modes=config.COUNTERFACTUAL_MODES,
        strengths=list(config.COUNTERFACTUAL_STRENGTHS.keys())
    )


@app.route("/report")
def report_page():
    """Generates and displays formal academic analysis report."""
    analysis = session.get("current_analysis")
    if not analysis:
        flash("No analysis data available to compile report. Please analyze an image first.", "warning")
        return redirect(url_for("predict_page"))

    report_payload = build_report_payload(
        image_filename=analysis["orig_filename"],
        detection_res=analysis["detection_res"],
        gradcam_res=analysis["gradcam_res"],
        cf_res=analysis.get("cf_res"),
        staging_res=analysis.get("staging_res")
    )

    # Save downloadable text report
    _, report_filename = save_downloadable_report(report_payload)

    return render_template(
        "report.html",
        report=report_payload,
        analysis=analysis,
        report_filename=report_filename
    )


@app.route("/report/download/<filename>")
def download_report(filename):
    """Secure endpoint to download compiled academic reports."""
    try:
        return send_from_directory(
            config.REPORTS_FOLDER, 
            filename, 
            as_attachment=True
        )
    except FileNotFoundError:
        flash("Requested report file could not be located.", "danger")
        return redirect(url_for("report_page"))


# ==========================================
# CHATBOT API ROUTES
# ==========================================
@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    """Processes user query and returns domain-specific academic response."""
    data = request.get_json(silent=True) or {}
    user_query = data.get("message", "")
    response_data = get_chatbot_response(user_query)
    return jsonify(response_data)


@app.route("/api/chatbot/suggestions", methods=["GET"])
def api_chatbot_suggestions():
    """Returns list of suggested questions."""
    return jsonify({"suggestions": get_suggested_queries()})


# ==========================================
# GLOBAL ERROR HANDLERS
# ==========================================
@app.errorhandler(413)
def request_entity_too_large(error):
    flash("The uploaded file exceeds the 16MB upload limit. Please upload a smaller image.", "danger")
    return redirect(url_for("predict_page"))


@app.errorhandler(404)
def page_not_found(error):
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    flash("An unexpected internal server error occurred. Please try again.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
