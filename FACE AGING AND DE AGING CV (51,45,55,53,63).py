"""
Face Aging & De-Aging Web Application
======================================
Flask backend for face aging/de-aging using computer vision techniques.

Best Models for Production:
---------------------------
1. SAM (Style-based Age Manipulation)  - SOTA GAN for fine-grained age control
2. FRAN (Face Re-Aging Network)        - Fast CNN-based, used in Disney VFX
3. HRFAE (High-Resolution Face Age Editing) - Identity-preserving autoencoder
4. Lifespan Age Transformation (LATS)  - Full lifespan GAN (0-70+ years)
5. SimSwap + Age Conditioning          - Identity + controllable aging

This implementation uses:
- OpenCV for face detection (Haar Cascade + DNN)
- PIL / scikit-image for image processing
- Wrinkle simulation, skin tone adjustments, facial geometry warping
- Optional hook to load pre-trained GAN weights (SAM/FRAN/HRFAE)
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps
import io
import os
import base64
import traceback
import math
import time
import json

# Optional heavy deps — app still starts without them
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠  opencv-python not yet installed — face detection unavailable until restart")

try:
    import scipy.ndimage as ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠  scipy not yet installed — wrinkle noise will use numpy fallback")

# ─────────────────────────────────────────────
#  Flask App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ─────────────────────────────────────────────
#  Utility Functions
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(pil_img: Image.Image, fmt: str = 'JPEG') -> str:
    """Convert PIL image to base64 string for JSON transport."""
    buf = io.BytesIO()
    if fmt == 'JPEG' and pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')
    pil_img.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def base64_to_image(b64str: str) -> Image.Image:
    """Decode base64 string to PIL Image and fix EXIF orientation."""
    data = base64.b64decode(b64str.split(',')[-1])  # strip data URI prefix
    img = Image.open(io.BytesIO(data))
    return ImageOps.exif_transpose(img)


def cv2_to_pil(cv2_img) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
#  Face Detection  (OpenCV 5.0 compatible)
# ─────────────────────────────────────────────
# OpenCV 5.x removed CascadeClassifier from the main module.
# Primary:  FaceDetectorYN  (YuNet DNN, built-in, no extra weights needed)
# Fallback: pure-PIL heuristic (whole image treated as single face)

_yunet_detector = None   # lazy singleton

def _get_yunet(width: int, height: int):
    """Return a cached FaceDetectorYN resized to the current frame dimensions."""
    global _yunet_detector
    if not CV2_AVAILABLE:
        return None
    try:
        if _yunet_detector is None:
            # FaceDetectorYN is bundled with opencv-python / opencv-python-headless >= 4.8
            _yunet_detector = cv2.FaceDetectorYN.create(
                model="",          # empty string → use built-in YuNet
                config="",
                input_size=(width, height),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5,
            )
        else:
            _yunet_detector.setInputSize((width, height))
        return _yunet_detector
    except Exception:
        return None


def detect_faces(cv2_img):
    """
    Detect faces and return list of (x, y, w, h) tuples.

    Strategy (OpenCV 5.0+):
      1. cv2.FaceDetectorYN  — YuNet DNN, built into cv2 >= 4.8, no extra files
      2. PIL-based heuristic — treats whole image as one face region
    """
    h_img, w_img = cv2_img.shape[:2]

    # ── Method 1: YuNet (preferred) ──────────────────────────────────────────
    if CV2_AVAILABLE:
        try:
            model_path = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")
            detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(w_img, h_img),
                score_threshold=0.55,
                nms_threshold=0.3,
                top_k=10,
            )
            _, faces = detector.detect(cv2_img)
            if faces is not None and len(faces) > 0:
                result = []
                for f in faces:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    result.append((x, y, w, h))
                return result
        except Exception as e:
            print(f"YuNet detection failed ({e}), using fallback")

    # ── Method 2: PIL face heuristic fallback ─────────────────────────────────
    # Treat the central 80 % of the image as the face region
    pad_x = int(w_img * 0.1)
    pad_y = int(h_img * 0.1)
    return [(pad_x, pad_y, w_img - 2 * pad_x, h_img - 2 * pad_y)]


# ─────────────────────────────────────────────
#  Age Estimation (Lightweight Heuristic)
# ─────────────────────────────────────────────
def _numpy_laplacian_var(gray_arr) -> float:
    """Pure-numpy Laplacian variance — no cv2 dependency."""
    # 3x3 Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    from PIL import Image as _PIL
    img_f = gray_arr.astype(np.float32)
    # Manual 2-D convolution via numpy stride tricks (small kernel — fast enough)
    h, w = img_f.shape
    lap = np.zeros_like(img_f)
    for dy in range(3):
        for dx in range(3):
            if kernel[dy, dx] == 0:
                continue
            lap[1:-1, 1:-1] += kernel[dy, dx] * img_f[dy:h-2+dy, dx:w-2+dx]
    return float(lap.var())


def estimate_age(face_roi_gray) -> int:
    """
    Rough age estimation using Laplacian texture variance.
    face_roi_gray: numpy uint8 array (grayscale face ROI)
    For production: use AgeNet (Caffe) or InsightFace age predictor.
    """
    if CV2_AVAILABLE:
        try:
            lap = cv2.Laplacian(face_roi_gray, cv2.CV_64F)
            texture = float(lap.var())
        except Exception:
            texture = _numpy_laplacian_var(face_roi_gray)
    else:
        texture = _numpy_laplacian_var(face_roi_gray)

    # Map texture variance → approximate age band
    if texture < 30:   return 20
    if texture < 80:   return 30
    if texture < 150:  return 40
    if texture < 300:  return 55
    return 65


# ─────────────────────────────────────────────
#  Core Aging / De-Aging Pipeline
# ─────────────────────────────────────────────
class FaceAgingProcessor:
    """
    Processes face images to simulate aging or de-aging effects.

    Production GAN Hook:
    ---------------------
    Replace `_apply_cv_aging()` with a GAN forward pass:
        from models.sam_model import SAMModel
        model = SAMModel.load('weights/sam.pt')
        output = model(face_tensor, target_age=age)
    """

    WRINKLE_REGIONS = {
        # (relative_y_start, relative_y_end, relative_x_start, relative_x_end, intensity)
        'forehead':     (0.08, 0.30, 0.15, 0.85, 1.0),
        'eye_corners_l': (0.30, 0.50, 0.05, 0.35, 0.8),
        'eye_corners_r': (0.30, 0.50, 0.65, 0.95, 0.8),
        'nose_mouth':   (0.60, 0.80, 0.25, 0.75, 0.6),
        'chin':         (0.80, 0.95, 0.20, 0.80, 0.5),
    }

    def process(self, pil_img: Image.Image, direction: str = 'age',
                target_age: int = 60) -> dict:
        """
        Main entry point. Uses State-of-the-Art GAN via HuggingFace Space.
        Falls back to local OpenCV heuristics if API fails.
        """
        try:
            import os
            from gradio_client import Client, handle_file
            
            temp_in = os.path.join(os.path.dirname(__file__), 'temp_in_gan.jpg')
            # Ensure RGB to save as JPEG
            pil_img_rgb = pil_img.convert('RGB')
            pil_img_rgb.save(temp_in, format='JPEG', quality=95)
            
            print("Calling High-Fidelity GAN for Face Aging...")
            client = Client("Robys01/Face-Aging")
            
            # Determine appropriate source age
            cv2_img = pil_to_cv2(pil_img)
            faces = detect_faces(cv2_img)
            est_age = 25
            if faces:
                x, y, w, h = faces[0]
                face_roi = cv2_img[y:y+h, x:x+w]
                if face_roi.size > 0:
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    est_age = estimate_age(gray_roi)

            # The GAN naturally handles both aging and de-aging based on target_age
            result_path = client.predict(
                image_path=handle_file(temp_in),
                source_age=float(est_age),
                target_age=float(target_age),
                api_name="/predict"
            )
            
            result_pil = Image.open(result_path)
            
            # Clean up temp file
            if os.path.exists(temp_in):
                os.remove(temp_in)
                
            return {
                'success'       : True,
                'original_b64'  : image_to_base64(pil_img),
                'result_b64'    : image_to_base64(result_pil),
                'faces_detected': len(faces) if faces else 1,
                'face_info'     : [{'bbox': faces[0] if faces else [0,0,cv2_img.shape[1],cv2_img.shape[0]], 'estimated_age': est_age}],
                'direction'     : direction,
                'target_age'    : target_age,
            }
        except Exception as e:
            print(f"GAN API failed ({e}), falling back to local OpenCV heuristics...")
            # Fallback to local CV heuristics
            cv2_img  = pil_to_cv2(pil_img)
            original = cv2_img.copy()
            faces    = detect_faces(cv2_img)

            if not faces:
                return {'success': False, 'error': 'No face detected in the image. '
                        'Please upload a clear frontal face photo.'}

            result_img = cv2_img.copy()
            face_info  = []

            for (x, y, w, h) in faces:
                pad  = int(max(w, h) * 0.15)
                x1   = max(0, x - pad)
                y1   = max(0, y - pad)
                x2   = min(cv2_img.shape[1], x + w + pad)
                y2   = min(cv2_img.shape[0], y + h + pad)

                face_roi  = cv2_img[y1:y2, x1:x2]
                gray_roi  = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                est_age   = estimate_age(gray_roi)

                if direction == 'age':
                    processed_roi = self._apply_aging(face_roi, target_age)
                else:
                    processed_roi = self._apply_deaging(face_roi, target_age)

                result_img[y1:y2, x1:x2] = processed_roi
                face_info.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'estimated_age': est_age
                })

            result_pil = cv2_to_pil(result_img)
            orig_pil   = cv2_to_pil(original)

            return {
                'success'       : True,
                'original_b64'  : image_to_base64(orig_pil),
                'result_b64'    : image_to_base64(result_pil),
                'faces_detected': len(faces),
                'face_info'     : face_info,
                'direction'     : direction,
                'target_age'    : target_age,
            }

    # ── Aging Effect ──────────────────────────────────────────────────────────
    def _apply_aging(self, face_roi, target_age: int):
        """
        Multi-stage aging pipeline:
        1. Skin texture degradation (wrinkles)
        2. Color / tone shift (yellowing, desaturation)
        3. Geometric morphing (sagging)
        5. Blending mask for natural look
        """
        age_factor = float(np.clip((target_age - 20) / 60.0, 0.0, 1.0))  # 0→1

        pil = cv2_to_pil(face_roi)

        # Step 1: Wrinkles
        pil = self._add_wrinkles(pil, age_factor)

        # Step 2: Skin tone shift
        pil = self._shift_skin_tone_age(pil, age_factor)

        # Step 3: Geometric sagging
        pil = self._apply_sagging(pil, age_factor)

        # Step 4: Slight blur (loss of skin detail)
        blur_r = max(0, age_factor * 1.2)
        if blur_r > 0.3:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_r))

        # Step 5: Reduce saturation
        enhancer = ImageEnhance.Color(pil)
        pil = enhancer.enhance(1.0 - age_factor * 0.35)

        # Step 6: Darken under-eye / shadow areas
        pil = self._add_age_spots(pil, age_factor)

        return pil_to_cv2(pil)

    # ── De-Aging Effect ───────────────────────────────────────────────────────
    def _apply_deaging(self, face_roi, target_age: int):
        """
        Multi-stage de-aging pipeline:
        1. Skin smoothing (bilateral + Gaussian)
        2. Brightness / clarity boost
        3. Saturation boost
        4. Slight geometric lifting
        """
        # How young we want to go (0 = very young, 1 = no change)
        youth_factor = float(np.clip((40 - target_age) / 30.0, 0.0, 1.0))

        pil = cv2_to_pil(face_roi)

        # Step 1: Skin smoothing via bilateral filter
        cv2_roi = pil_to_cv2(pil)
        smooth  = cv2.bilateralFilter(cv2_roi, d=9,
                                       sigmaColor=int(60 + youth_factor * 40),
                                       sigmaSpace=int(60 + youth_factor * 40))
        # Blend with original
        alpha = 0.4 + youth_factor * 0.4
        blended = cv2.addWeighted(smooth, alpha, cv2_roi, 1 - alpha, 0)
        pil     = cv2_to_pil(blended)

        # Step 2: Brightness lift
        enhancer = ImageEnhance.Brightness(pil)
        pil = enhancer.enhance(1.0 + youth_factor * 0.12)

        # Step 3: Contrast clarity
        enhancer = ImageEnhance.Contrast(pil)
        pil = enhancer.enhance(1.0 + youth_factor * 0.08)

        # Step 4: Saturation boost (youthful, vibrant skin)
        enhancer = ImageEnhance.Color(pil)
        pil = enhancer.enhance(1.0 + youth_factor * 0.25)

        # Step 5: Sharpness (youthful skin definition)
        enhancer = ImageEnhance.Sharpness(pil)
        pil = enhancer.enhance(1.0 + youth_factor * 0.3)

        # Step 6: Geometric lift (reverse sagging)
        pil = self._apply_lifting(pil, youth_factor)

        return pil_to_cv2(pil)

    # ── Wrinkle Simulation ────────────────────────────────────────────────────
    def _add_wrinkles(self, pil_img: Image.Image, factor: float) -> Image.Image:
        """Deepen existing facial lines and wrinkles using high-frequency edge extraction."""
        if factor < 0.05:
            return pil_img
            
        cv_img = pil_to_cv2(pil_img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Extract high frequency details (dark lines, folds, pores)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # diff is positive where the original image is darker than the local neighborhood
        diff = cv2.subtract(blurred, gray)
        
        # Amplify the dark lines based on the age factor
        amp = factor * 4.0
        wrinkle_mask = cv2.multiply(diff, amp)
        
        # Smooth the mask slightly so the wrinkles don't look like sharp noise
        wrinkle_mask = cv2.GaussianBlur(wrinkle_mask, (3, 3), 0)
        
        cv_img_f = cv_img.astype(np.float32)
        mask_3d = cv2.merge([wrinkle_mask, wrinkle_mask, wrinkle_mask]).astype(np.float32)
        
        h, w = cv_img_f.shape[:2]
        
        # Darken the image where existing lines are, ONLY in specific facial regions
        for region_name, (ys, ye, xs, xe, region_factor) in self.WRINKLE_REGIONS.items():
            y0, y1 = int(h * ys), int(h * ye)
            x0, x1 = int(w * xs), int(w * xe)
            
            # Apply regional weight
            region_mask = mask_3d[y0:y1, x0:x1] * region_factor
            cv_img_f[y0:y1, x0:x1] -= region_mask
            
        cv_img_f = np.clip(cv_img_f, 0, 255).astype(np.uint8)
        
        return cv2_to_pil(cv_img_f)

    # ── Skin Tone Shift (Aging) ────────────────────────────────────────────────
    def _shift_skin_tone_age(self, pil_img: Image.Image, factor: float) -> Image.Image:
        """Shift skin toward yellower, darker, duller tones."""
        arr = np.array(pil_img).astype(np.float32)

        # Red channel slight increase (ruddy / aged)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + factor * 8, 0, 255)
        # Green channel slight decrease
        arr[:, :, 1] = np.clip(arr[:, :, 1] - factor * 5, 0, 255)
        # Blue channel decrease (yellowing)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - factor * 15, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    # ── Sagging Geometry ──────────────────────────────────────────────────────
    def _apply_sagging(self, pil_img: Image.Image, factor: float) -> Image.Image:
        """Simulate gravity sagging using vertical pixel displacement."""
        if factor < 0.15:
            return pil_img
        w, h  = pil_img.size
        arr   = np.array(pil_img).astype(np.float32)
        result= arr.copy()

        sag_max = int(factor * h * 0.04)  # Natural sagging effect
        if sag_max < 1:
            return pil_img

        for row in range(h):
            rel     = row / h
            # Sag increases toward bottom of face
            sag     = int(sag_max * math.sin(rel * math.pi) * rel)
            src_row = min(row + sag, h - 1)
            result[row] = arr[src_row]

        return Image.fromarray(result.astype(np.uint8))

    # ── Age Spots ─────────────────────────────────────────────────────────────
    def _add_age_spots(self, pil_img: Image.Image, factor: float) -> Image.Image:
        """Add subtle age/liver spots on the skin."""
        if factor < 0.4:
            return pil_img
        w, h   = pil_img.size
        draw   = ImageDraw.Draw(pil_img.copy())
        arr    = np.array(pil_img).astype(np.float32)

        n_spots = int(factor * 20)  # more spots
        rng     = np.random.default_rng(seed=42)

        for _ in range(n_spots):
            sx    = rng.integers(int(w * 0.1), int(w * 0.9))
            sy    = rng.integers(int(h * 0.2), int(h * 0.85))
            sr    = rng.integers(4, 12)  # slightly larger
            y0, y1 = max(0, sy-sr), min(h, sy+sr)
            x0, x1 = max(0, sx-sr), min(w, sx+sr)
            # Darken the spot more aggressively
            arr[y0:y1, x0:x1] *= (1.0 - factor * 0.4)

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # ── Geometric Lifting (De-Aging) ──────────────────────────────────────────
    def _apply_lifting(self, pil_img: Image.Image, factor: float) -> Image.Image:
        """Reverse sagging — lift features upward slightly."""
        if factor < 0.15:
            return pil_img
        w, h   = pil_img.size
        arr    = np.array(pil_img).astype(np.float32)
        result = arr.copy()

        lift_max = int(factor * h * 0.03)
        if lift_max < 1:
            return pil_img

        for row in range(h):
            rel      = row / h
            lift     = int(lift_max * math.sin(rel * math.pi) * (1 - rel))
            src_row  = max(0, row - lift)
            result[row] = arr[src_row]

        return Image.fromarray(result.astype(np.uint8))


# Singleton processor
processor = FaceAgingProcessor()


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/process', methods=['POST'])
def process_image():
    """
    Endpoint: POST /api/process
    Body (JSON):
        image_b64  : str   – base64 encoded image (with or without data URI prefix)
        direction  : str   – 'age' | 'deage'
        target_age : int   – desired output age (10–80)
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body received.'}), 400

        image_b64  = data.get('image_b64', '')
        direction  = data.get('direction', 'age')
        target_age = int(data.get('target_age', 60))

        if not image_b64:
            return jsonify({'success': False, 'error': 'No image data provided.'}), 400

        if direction not in ('age', 'deage'):
            return jsonify({'success': False, 'error': 'direction must be "age" or "deage".'}), 400

        target_age = int(np.clip(target_age, 10, 80))

        # Decode
        pil_img = base64_to_image(image_b64)

        # Resize if too large (speed)
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            ratio   = max_dim / max(pil_img.size)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img  = pil_img.resize(new_size, Image.LANCZOS)

        start  = time.time()
        result = processor.process(pil_img, direction=direction, target_age=target_age)
        elapsed = round(time.time() - start, 2)
        result['processing_time'] = elapsed

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Alternative endpoint for multipart form upload.
    Returns base64 of the uploaded image for preview.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False,
                        'error': f'Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    data    = file.read()
    pil_img = Image.open(io.BytesIO(data))
    b64     = image_to_base64(pil_img)

    return jsonify({'success': True, 'image_b64': f'data:image/jpeg;base64,{b64}',
                    'width': pil_img.width, 'height': pil_img.height})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status'    : 'ok',
        'models_info': {
            'current'       : 'OpenCV + PIL pipeline (CV heuristic)',
            'recommended'   : ['SAM', 'FRAN', 'HRFAE', 'LATS'],
            'face_detector' : 'Haar Cascade (DNN fallback if weights present)',
        }
    })


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Face Aging & De-Aging Web Application")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
