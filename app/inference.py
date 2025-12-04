#Kody Graham
#12/02/2025
#This class will load my trained model and analyze the images with grad-cam

from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf

from gradcam import make_gradcam_heatmap, overlay_heatmap_on_image
from model.model_def import build_tamper_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "tamper_detector.h5"
CLASS_MAP_PATH = BASE_DIR / "model" / "class_indices.json"

IMG_SIZE = (224, 224)

TAMPER_LABELS = ["original", "tampered"]
TYPE_LABELS = ["original", "jpeg", "blur", "noise", "copy_move", "splice", "inpaint"]


class TamperDetector:
    def __init__(self) -> None:
        self.is_trained: bool = False


        if MODEL_PATH.exists():
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.is_trained = True
        else:
            self.model = build_tamper_model(
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                num_tamper_classes=len(TAMPER_LABELS),
                num_type_classes=len(TYPE_LABELS),
            )
            self.is_trained = False

        self.class_indices = {
            "tamper": {i: name for i, name in enumerate(TAMPER_LABELS)},
            "type": { i: name for i, name in enumerate(TYPE_LABELS)},
        }

        if CLASS_MAP_PATH.exists():
            try:
                with open(CLASS_MAP_PATH, "r", encoding= "utf-8") as f:
                    data = json.load(f)
                for key in ("tamper", "type"):
                    if key in data:
                        self.class_indices[key] = { int(k): v for k, v in data[key].items() }

            except Exception as e:
                print(f"Warning: failed to load class indices: {e}")

    def preprocess(self,image_bgr: np.ndarray) -> np.ndarray:

        img_resized = cv2.resize(image_bgr, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = img_rgb.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict_raw(self, x: np.ndarray):

        predicts = self.model.predict(x, verbose=0)

        if isinstance(predicts, dict):
            tamper_props = predicts["tamper_output"]
            type_probs = predicts["type_output"]

        else:
            tamper_props, type_probs = predicts

        return tamper_props, type_probs

    def analyze_image(self, image_bgr: np.ndarray) -> dict:

        x = self.preprocess(image_bgr)

        tamper_probs, type_probs = self.predict_raw(x)

        tamper_idx = int(np.argmax(tamper_probs[0]))
        tamper_label = self.class_indices["tamper"].get(tamper_idx, "unknown")
        tamper_conf = float(tamper_probs[0][tamper_idx])

        type_idx = int(np.argmax(type_probs[0]))
        type_label = self.class_indices["type"].get(type_idx, "unknown")
        type_conf = float(type_probs[0][type_idx])

        if tamper_label == "tampered":
            heatmap = make_gradcam_heatmap(x, self.model, last_conv_layer_name="last_conv", pred_index = tamper_idx, )
            overlay = overlay_heatmap_on_image(image_bgr, heatmap, alpha = .5)

        else:
            heatmap= None
            overlay = None

        return {"tamper_label": tamper_label, "tamper_confidence": tamper_conf, "type_label": type_label, "type_confidence": type_conf, "heatmap": heatmap, "overlay_bgr": overlay,}

