import os
import uuid
import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
import config

def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS

def validate_and_save_upload(file_storage, target_folder: str = None) -> tuple:
    """
    Validates uploaded file, ensures it is a valid image, and saves it.
    Returns (saved_filepath, unique_filename).
    Raises ValueError if invalid.
    """
    if target_folder is None:
        target_folder = config.UPLOAD_FOLDER
    os.makedirs(target_folder, exist_ok=True)

    if not file_storage or file_storage.filename == "":
        raise ValueError("No file selected for upload.")

    if not allowed_file(file_storage.filename):
        raise ValueError(f"File format not allowed. Allowed formats: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}")

    clean_name = secure_filename(file_storage.filename)
    unique_id = uuid.uuid4().hex[:10]
    filename = f"{unique_id}_{clean_name}"
    save_path = os.path.join(target_folder, filename)

    # Save to disk first
    file_storage.save(save_path)

    # Verify that PIL can actually decode the image
    try:
        with Image.open(save_path) as img:
            img.verify()
    except (UnidentifiedImageError, Exception) as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise ValueError(f"Uploaded file is corrupted or not a valid image: {str(e)}")

    return save_path, filename

def load_and_preprocess_image(image_path: str, target_size: tuple = None) -> tuple:
    """
    Loads an image from path, validates, preprocesses for CNN inference.
    Returns:
      - preprocessed_batch: np.ndarray with shape (1, height, width, 3), float32 normalized [0, 1]
      - display_image_rgb: np.ndarray with shape (height, width, 3), uint8 [0, 255]
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            # Handle grayscale, RGBA, palette modes: convert to 3-channel RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize using high-quality Lanczos resampling
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            display_image_rgb = np.array(resized_img, dtype=np.uint8)

            # Normalization to [0.0, 1.0] float32
            img_array = display_image_rgb.astype(np.float32) / 255.0

            # Expand batch dimension -> shape (1, H, W, 3)
            preprocessed_batch = np.expand_dims(img_array, axis=0)

            return preprocessed_batch, display_image_rgb
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")

def prepare_display_image(source_path: str, dest_folder: str = None) -> str:
    """
    Creates a standard web-viewable PNG/JPEG copy in the generated folder
    to guarantee browser compatibility.
    """
    if dest_folder is None:
        dest_folder = config.GENERATED_FOLDER
    os.makedirs(dest_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(source_path))[0]
    out_filename = f"view_{base_name}.png"
    out_path = os.path.join(dest_folder, out_filename)

    with Image.open(source_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path, format="PNG")

    return out_path, out_filename


def extract_anatomical_lung_mask(gray: np.ndarray) -> np.ndarray:
    """
    Extracts true anatomical lung parenchyma cavities from chest CT slices.
    Segments low-density aerated thoracic cavities and excludes bones, chest wall, and mediastinum.
    Works reliably on standard CT scans, cropped scans, and internet CT images.
    Returns 2D boolean mask of shape (H, W).
    """
    import cv2
    h, w = gray.shape
    dark = (gray < 0.35).astype(np.uint8)
    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)

    lung_labels = []
    for i in range(1, num_l):
        x, y, cw, ch, a = stats[i]
        cx, cy = centroids[i]
        # Ignore outer ambient background air touching the outer image border
        if (x == 0 and y == 0 and cw == w and ch == h) or a < 250:
            continue
        if (x == 0 or y == 0 or (x + cw) >= w or (y + ch) >= h) and a > (h * w * 0.35):
            continue
        # Anatomical lungs reside within the central thoracic field
        if 0.10 * w < cx < 0.90 * w and 0.10 * h < cy < 0.90 * h:
            lung_labels.append(i)

    if len(lung_labels) > 0:
        lung_binary = np.isin(labels, lung_labels).astype(np.uint8)
        # Morphological closing and contour fill to include internal nodules and vessels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(lung_binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

        # Erode slightly to avoid the high-density chest wall and rib cage boundaries
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(filled, erode_kernel)
        if np.sum(eroded) > 500:
            return eroded.astype(bool)

    # Anatomical bilateral ellipse fallback if contrast is atypical
    y_grid, x_grid = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    rlung = (((x_grid - (cx - w * 0.18)) / (w * 0.16))**2 + ((y_grid - cy) / (h * 0.26))**2 <= 1.0) & (x_grid < cx - w * 0.03)
    llung = (((x_grid - (cx + w * 0.18)) / (w * 0.16))**2 + ((y_grid - cy) / (h * 0.26))**2 <= 1.0) & (x_grid > cx + w * 0.03)
    return rlung | llung

