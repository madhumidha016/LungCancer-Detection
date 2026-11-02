import os
import shutil
import uuid
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
from case_retrieval import retrieve_similar_cases, ensure_case_images, CLINICAL_CASE_LIBRARY
from progression_simulator import (
    analyze_longitudinal_progression, ensure_progression_demo_images
)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# In-memory storage caches to keep client cookies lightweight (< 200 bytes)
ANALYSIS_CACHE = {}
PROGRESSION_CACHE = {}

def get_sid() -> str:
    """Retrieves or initializes unique session identifier."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]

def get_current_analysis():
    """Retrieves current analysis from server memory cache with fallback to cookie."""
    sid = session.get("sid")
    if sid and sid in ANALYSIS_CACHE:
        return ANALYSIS_CACHE[sid]
    return session.get("current_analysis")

def get_current_progression():
    """Retrieves progression data from server memory cache with fallback to cookie."""
    sid = session.get("sid")
    if sid and sid in PROGRESSION_CACHE:
        return PROGRESSION_CACHE[sid]
    return session.get("progression_analysis")


# Ensure all workspace directories and initial demo assets exist
for folder in [
    config.UPLOAD_FOLDER, 
    config.GENERATED_FOLDER, 
    config.REPORTS_FOLDER, 
    config.MODELS_FOLDER,
    config.CASES_FOLDER,
    config.PROGRESSION_FOLDER
]:
    os.makedirs(folder, exist_ok=True)

# Generate baseline reference images if missing
ensure_case_images()
ensure_progression_demo_images()

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

        # Staging check
        staging_model = get_staging_model()
        staging_res = predict_staging(
            staging_model=staging_model,
            preprocessed_batch=preprocessed_batch,
            cancer_prob=detection_res["cancer_probability"]
        )

        # FEATURE 1: Similar Case Retrieval System
        similar_cases = retrieve_similar_cases(
            preprocessed_batch=preprocessed_batch,
            detection_res=detection_res,
            staging_res=staging_res,
            top_k=config.SIMILAR_CASES_TOP_K
        )

        # Cache in server memory to keep client session cookies ultra-light
        sid = get_sid()
        full_analysis = {
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
            "similar_cases": similar_cases,
            "cf_res": None,
            "chart_filename": None
        }
        ANALYSIS_CACHE[sid] = full_analysis

        # Minimal cookie representation
        session["current_analysis"] = {
            "image_path": saved_path,
            "orig_filename": orig_filename,
            "display_filename": display_filename,
            "detection_res": detection_res,
            "staging_res": staging_res
        }

        # Keep heatmap array in memory or temporary scratch for counterfactual
        if gradcam_res.get("heatmap_array") is not None:
            heat_temp = os.path.join(config.GENERATED_FOLDER, f"temp_heat_{orig_filename}.npy")
            import numpy as np
            np.save(heat_temp, gradcam_res["heatmap_array"])
            session["heatmap_temp_path"] = heat_temp

        flash(
            f"Inference complete: Predicted {detection_res['predicted_class']} "
            f"({detection_res['confidence_pct']}%) • Retrieved {len(similar_cases)} Similar Clinical Cases", 
            "success"
        )
        return redirect(url_for("result_page"))

    except ValueError as ve:
        flash(str(ve), "danger")
        return redirect(url_for("predict_page"))
    except Exception as e:
        flash(f"An error occurred during image processing: {str(e)}", "danger")
        return redirect(url_for("predict_page"))


@app.route("/result")
def result_page():
    """Displays detection metrics, Grad-CAM heatmap, staging, and similar cases."""
    analysis = get_current_analysis()
    if not analysis:
        flash("No active analysis found. Please upload a CT scan image first.", "info")
        return redirect(url_for("predict_page"))

    return render_template(
        "result.html",
        analysis=analysis,
        detection=analysis["detection_res"],
        gradcam=analysis.get("gradcam_res", {}),
        staging=analysis.get("staging_res", {}),
        similar_cases=analysis.get("similar_cases", [])
    )


@app.route("/similar-cases")
def similar_cases_page():
    """Dedicated interface for the Similar Case Retrieval Library."""
    analysis = get_current_analysis()
    retrieved = analysis.get("similar_cases", []) if analysis else []
    return render_template(
        "similar_cases.html",
        analysis=analysis,
        retrieved_cases=retrieved,
        all_library_cases=CLINICAL_CASE_LIBRARY
    )


# ==========================================
# FEATURE 2: DISEASE PROGRESSION SIMULATOR
# ==========================================
@app.route("/progression", methods=["GET", "POST"])
def progression_page():
    """
    Longitudinal Disease Progression Simulator.
    Allows uploading multiple CT scans taken at different time intervals,
    quantifies volumetric tumor growth kinetics, generates progression graphs,
    and predicts possible next-stage progression.
    """
    if request.method == "POST":
        try:
            # Check how many timepoint inputs were provided
            files = request.files.getlist("progression_files")
            labels = request.form.getlist("timepoint_labels")
            months = request.form.getlist("timepoint_months")

            valid_entries = []
            for idx, file in enumerate(files):
                if file and file.filename != "":
                    label = labels[idx].strip() if idx < len(labels) and labels[idx].strip() else f"Timepoint {idx+1}"
                    try:
                        month_val = float(months[idx]) if idx < len(months) and months[idx].strip() else float(idx * 3)
                    except ValueError:
                        month_val = float(idx * 3)
                    
                    saved_path, _ = validate_and_save_upload(file)
                    valid_entries.append({
                        "label": label,
                        "month": month_val,
                        "image_path": saved_path
                    })

            if len(valid_entries) < config.PROGRESSION_TIMEPOINTS_MIN:
                flash(
                    f"Please provide at least {config.PROGRESSION_TIMEPOINTS_MIN} CT scans with time intervals "
                    f"to compute growth velocity, or click 'Load Clinical Demo Follow-up Case'.", 
                    "warning"
                )
                return redirect(url_for("progression_page"))

            # Execute Longitudinal Progression Analysis
            progression_data = analyze_longitudinal_progression(valid_entries)
            sid = get_sid()
            PROGRESSION_CACHE[sid] = progression_data
            session["has_progression"] = True

            flash(
                f"Progression Analysis Complete: RECIST 1.1 Status: {progression_data['recist_category']} • "
                f"Predicted Next Stage: {progression_data['forecast']['next_stage_name']}",
                "success"
            )
            return redirect(url_for("progression_page"))

        except Exception as e:
            flash(f"Progression analysis error: {str(e)}", "danger")
            return redirect(url_for("progression_page"))

    progression_data = get_current_progression()
    return render_template(
        "progression.html",
        progression=progression_data
    )


@app.route("/progression/demo")
def progression_demo():
    """Instant 1-click loading of calibrated 3-timepoint longitudinal progression case."""
    try:
        demo_paths = ensure_progression_demo_images()
        timepoints = [
            {"label": "Baseline (Day 0)", "month": 0.0, "image_path": demo_paths[0]},
            {"label": "Follow-up (Month 3)", "month": 3.0, "image_path": demo_paths[1]},
            {"label": "Follow-up (Month 6)", "month": 6.0, "image_path": demo_paths[2]}
        ]
        progression_data = analyze_longitudinal_progression(timepoints)
        sid = get_sid()
        PROGRESSION_CACHE[sid] = progression_data
        session["has_progression"] = True

        flash(
            "Loaded Clinical Demo Case: 6-Month Longitudinal Adenocarcinoma Tracking. "
            "Visualizing volumetric expansion, RECIST evaluation, and next-stage forecast.",
            "info"
        )
        return redirect(url_for("progression_page"))
    except Exception as e:
        flash(f"Error loading progression demo: {str(e)}", "danger")
        return redirect(url_for("progression_page"))


@app.route("/progression/reset")
def progression_reset():
    """Resets the progression session."""
    sid = session.get("sid")
    if sid and sid in PROGRESSION_CACHE:
        del PROGRESSION_CACHE[sid]
    session.pop("has_progression", None)
    flash("Progression analysis session cleared. Ready for new CT timepoint scans.", "info")
    return redirect(url_for("progression_page"))


# ==========================================
# COUNTERFACTUAL WHAT-IF ROUTES
# ==========================================
@app.route("/counterfactual", methods=["GET", "POST"])
def counterfactual_page():
    """Counterfactual What-If interface for causal perturbation analysis."""
    analysis = get_current_analysis()
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

            # Update cache
            analysis["cf_res"] = cf_res
            analysis["chart_filename"] = chart_filename
            sid = get_sid()
            ANALYSIS_CACHE[sid] = analysis

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
    analysis = get_current_analysis()
    if not analysis:
        flash("No analysis data available to compile report. Please analyze an image first.", "warning")
        return redirect(url_for("predict_page"))

    report_payload = build_report_payload(
        image_filename=analysis["orig_filename"],
        detection_res=analysis["detection_res"],
        gradcam_res=analysis.get("gradcam_res"),
        cf_res=analysis.get("cf_res"),
        staging_res=analysis.get("staging_res"),
        similar_cases=analysis.get("similar_cases", [])
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
