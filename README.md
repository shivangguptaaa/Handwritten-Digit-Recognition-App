# Handwritten Digit Recognition

Interactive MNIST digit classifier built with a TensorFlow/Keras CNN, OpenCV preprocessing, Flask, and HTML5 Canvas. Draw any digit (`0-9`) on the canvas (or try one of the quick sample strokes) to inspect the predicted digit, confidence, inference latency, 10-class probability distribution, and the exact `28x28` preprocessed image passed into the model.

## Architecture & Preprocessing

- **Model (`train_model.py`, `models/mnist_cnn.h5`):** 4-layer `Conv2D` network (`32` and `64` filters) with `BatchNormalization`, `MaxPooling2D`, `Dropout` (`0.25` / `0.5`), a `256`-unit dense layer, and a `10`-class softmax head trained with `EarlyStopping`.
- **OpenCV Normalization (`app.py`):**
  1. Safely decodes RGBA/RGB/grayscale strokes and auto-inverts dark-on-light images using border luminance.
  2. Crops tightly around the digit and scales the longest dimension to `20px` while preserving aspect ratio (preventing thin digits like `1` or `7` from stretching).
  3. Centers inside a `28x28` frame using center-of-mass shifting and applies `(3x3)` Gaussian smoothing.
- **API Endpoints:**
  - `POST /predict` — returns `digit`, `confidence`, `probabilities`, base64 `preview` (`28x28`), and `latency_ms` (returns `null` cleanly for blank canvases).
  - `GET /health` — simple JSON health check.

## Structure

```
handwritten-digit-recognition/
├── app.py                  # Flask server + OpenCV preprocessing + inference endpoints
├── train_model.py          # MNIST CNN training script
├── models/
│   └── mnist_cnn.h5        # Saved Keras CNN weights
├── templates/
│   └── index.html          # Canvas UI + 28x28 preview + class probabilities
├── Dockerfile              # Container build for Docker / Render deployment
├── render.yaml             # Render free-tier service blueprint
└── requirements.txt
```

## Run Locally

`tensorflow-cpu` supports Python `3.9-3.12` (or use Docker for any host Python version).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
# http://localhost:7860
```

Or with Docker:

```bash
docker build -t digit-recognizer .
docker run -p 7860:7860 digit-recognizer
```

## Free Deployment (Render + GitHub)

Includes `Dockerfile` and `render.yaml` for deploying on Render's free Docker tier (`GET /health` health check configured). Connect the GitHub repository on [render.com](https://render.com) via **New + -> Web Service** using the **Docker** runtime on the **Free** plan.
