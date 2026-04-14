import numpy as np
from tensorflow.keras.models import load_model

# Load once (IMPORTANT)
model = None
class_labels = None

def load_artifacts():
    global model, class_labels
    if model is None:
        model = load_model("../ml/models/plant_disease_model.h5")
        class_labels = np.load("../ml/models/class_labels.npy", allow_pickle=True)

def predict_image(image):
    load_artifacts()

    preds = model.predict(image)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return {
        "disease": str(class_labels[idx]),
        "confidence": confidence
    }
