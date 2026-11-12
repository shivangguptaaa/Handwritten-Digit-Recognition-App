from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.h5"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# Load model once at startup
model = load_model(str(MODEL_PATH))


def _encode_preview_png(img_28x28: np.ndarray) -> str:
    """Encode a normalized (28, 28) float32 array as a base64 PNG data URI."""
    uint8_img = np.clip(img_28x28 * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", uint8_img)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def preprocess_image(img_data: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode base64 canvas image into a centered, aspect-preserved (1, 28, 28, 1) tensor."""
    if not img_data or not isinstance(img_data, str):
        return None

    cleaned = re.sub(r"^data:image/.+;base64,", "", img_data)
    try:
        img_bytes = base64.b64decode(cleaned)
    except Exception:
        return None

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    if img_array.size == 0:
        return None

    image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    # Convert to grayscale safely regardless of channel count
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        bgr_gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if np.any(alpha > 0.1) and np.mean(alpha) < 0.99 and np.mean(bgr_gray[alpha > 0.1]) < 60:
            gray = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If the user uploaded a dark-ink-on-white-paper image, invert before cropping
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.mean(border_pixels)) > 127.0:
        gray = 255 - gray

    # Suppress faint compression noise and skip blank canvas
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_TOZERO)
    if int(np.sum(thresh)) < 1000:
        return None

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y : y + h, x : x + w]

    # Preserve aspect ratio: fit longer dimension into 20px (MNIST convention)
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in a 28x28 frame, then gently refine with center-of-mass alignment
    canvas = np.zeros((28, 28), dtype=np.float32)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized.astype(np.float32) / 255.0

    total_mass = float(np.sum(canvas))
    if total_mass > 0:
        ys, xs = np.indices((28, 28))
        cy = float(np.sum(ys * canvas) / total_mass)
        cx = float(np.sum(xs * canvas) / total_mass)
        shift_x = int(np.clip(round(13.5 - cx), -3, 3))
        shift_y = int(np.clip(round(13.5 - cy), -3, 3))
        if shift_x != 0 or shift_y != 0:
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, M, (28, 28), borderValue=0.0)

    smoothed = cv2.GaussianBlur(canvas, (3, 3), 0.45)
    peak = float(np.max(smoothed))
    if peak > 0:
        smoothed = np.clip(smoothed / peak, 0.0, 1.0)

    processed = smoothed.reshape(1, 28, 28, 1)
    return processed, smoothed


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "mnist_cnn"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    img_data = data.get("image")
    if not img_data:
        return jsonify({"digit": None, "confidence": 0, "probabilities": [], "preview": None}), 400

    result = preprocess_image(img_data)
    if result is None:
        return jsonify({"digit": None, "confidence": 0, "probabilities": [], "preview": None})

    processed, preview_28 = result

    t0 = time.perf_counter()
    prediction = model.predict(processed, verbose=0)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    probs = prediction[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    probabilities = [float(p) for p in probs]

    return jsonify(
        {
            "digit": digit,
            "confidence": round(confidence * 100, 1),
            "probabilities": probabilities,
            "preview": _encode_preview_png(preview_28),
            "latency_ms": latency_ms,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
