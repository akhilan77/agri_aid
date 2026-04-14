from pathlib import Path

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "ml" / "models" / "plant_disease_model.h5"
LABELS_PATH = REPO_ROOT / "ml" / "models" / "class_labels.npy"

# Load once (IMPORTANT)
model = None
class_labels = None


def load_artifacts():
    global model, class_labels
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not LABELS_PATH.exists():
            raise FileNotFoundError(f"Class labels file not found at {LABELS_PATH}")

        model = load_model(str(MODEL_PATH))
        class_labels = np.load(str(LABELS_PATH), allow_pickle=True)


def preprocess_image(image):
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    return preprocess_input(image)


def predict_image(image):
    load_artifacts()
    processed = preprocess_image(image)

    preds = model.predict(processed, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "disease": str(class_labels[idx]),
        "confidence": confidence
    }
