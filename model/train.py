#Kody Graham
#12/01/2025
#This class will load the images, auto generate tampered variants, and trains my CNN.

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from model.model_def import build_tamper_model

#Reproduce
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = BASE_DIR / "data" / "original"
SAVE_DIR = BASE_DIR / "saved_models"
CLASS_MAP_PATH = BASE_DIR / "model" / "class_indices.json"
MODEL_PATH = SAVE_DIR / "tamper_detector.h5"

LOG_DIR = BASE_DIR / "logs" / "tamper_training"

IMG_SIZE = (224, 224)

#Labels
TAMPER_LABELS = ["original", "tampered"]
TYPE_LABELS = ["original", "jpeg", "blur", "noise", "copy_move", "splice", "inpaint"]

def ensure_dirs():
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def load_original_images() -> list[np.ndarray]:
    if not ORIGINAL_DIR.exists():
        return []

    images = []
    for ext in ("*.jpg","*.jpeg", "*.png"):
        for path in ORIGINAL_DIR.glob(ext):
            img = cv2.imread(str(path))
            if img is None:
                continue
            images.append(img)

    return images

def preprocess_for_model(image_bgr: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(image_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype("float32")/ 255.0
    return img_array

def random_rect(h: int, w: int):
    x1 = random.randint(0, max(0, w - w // 2))
    y1 = random.randint(0, max(0, h - h // 2))
    x2 = random.randint(x1 + max(4, w // 8), min(w, x1 + w // 2))
    y2 = random.randint(y1 + max(4, h // 8), min(h, y1 + h // 2))

    return x1, y1, x2, y2

def tamper_jpeg(image_bgr: np.ndarray) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
    success, enc = cv2.imencode('.jpg', image_bgr, encode_param)
    if not success:
        return image_bgr.copy()
    dec =cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

def tamper_blur(image_bgr: np.ndarray) -> np.ndarray:
    h,w = image_bgr.shape[:2]
    img = image_bgr.copy()

    #Random rectangles
    x1, y1, x2, y2 = random_rect(h, w)

    roi = img[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi, (15,15), 0)
    img[y1:y2, x1:x2] = roi_blur
    return img

def tamper_noise(image_bgr: np.ndarray) -> np.ndarray:

    h, w = image_bgr.shape[:2]
    img = image_bgr.copy()

    x1, y1, x2, y2 = random_rect(h, w)

    roi = img[y1:y2, x1:x2].astype("float32")
    noise = np.random.normal(loc=0.0,scale=30.0,size=roi.shape)
    roi_noisy= np.clip(roi + noise, 0, 255).astype("uint8")
    img[y1:y2, x1:x2] = roi_noisy
    return img

def tamper_copy_move(image_bgr: np.ndarray) -> np.ndarray:
    h,w = image_bgr.shape[:2]
    img = image_bgr.copy()

    #Source path
    src_x1, src_y1, src_x2, src_y2 = random_rect(h,w)
    patch = img[src_y1:src_y2, src_x1:src_x2].copy()
    ph, pw = patch.shape[:2]

    dst_x1 = random.randint(0, w - pw)
    dst_y1 = random.randint(0, h - ph)
    dst_x2 = dst_x1 + pw
    dst_y2 = dst_y1 + ph

    img[dst_y1:dst_y2, dst_x1:dst_x2] = patch
    return img

def tamper_splice(target_bgr: np.ndarray, donor_bgr: np.array) -> np.ndarray:
    h,w = target_bgr.shape[:2]
    img = target_bgr.copy()

    donor = donor_bgr.copy()
    donor = cv2.resize(donor, (w,h))

    src_x1, src_y1, src_x2, src_y2 = random_rect(h, w)
    patch = donor[src_y1:src_y2, src_x1:src_x2].copy()
    ph, pw = patch.shape[:2]

    dst_x1 = random.randint(0, max(0, w - pw))
    dst_y1 = random.randint(0, max(0, h - ph))
    dst_x2 = dst_x1 + pw
    dst_y2 = dst_y1 + ph

    patch_float = patch.astype("float32")
    alpha = .8 + .4 * random.random()
    beta = random.randint(-15,15)
    patch_jitter = np.clip(alpha * patch_float + beta, 0, 255).astype("uint8")

    img[dst_y1:dst_y2, dst_x1:dst_x2] = patch_jitter
    return img

def tamper_inpaint(image_bgr: np.ndarray) -> np.ndarray:
    h,w = image_bgr.shape[:2]
    img = image_bgr.copy()

    x1, y1, x2, y2 = random_rect(h, w)
    mask = np.zeros((h,w), dtype="uint8")
    mask[y1:y2, x1:x2] = 255

    inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags = cv2.INPAINT_TELEA)
    return inpainted

class LiveTrainingPlot(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()

        self.epochs = []
        self.tamper_loss = []
        self.tamper_acc = []
        self.val_tamper_loss = []
        self.val_tamper_acc = []

        plt.ion()

        self.fig, self.axes = plt.subplots(2,2, figsize=(12,8), facecolor="black", constrained_layout=True)
        (self.ax_loss_train, self.ax_loss_val), (self.ax_acc_train, self.ax_acc_val) = self.axes  # NEW

        for ax in (self.ax_loss_train, self.ax_loss_val, self.ax_acc_train, self.ax_acc_val):

            ax.set_facecolor("black")
            ax.tick_params(colors ="red")
            ax.xaxis.label.set_color("red")
            ax.yaxis.label.set_color("red")

            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)


        self.init_axes()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def init_axes(self):

        self.ax_loss_train.set_title("Train Loss", color="red")
        self.ax_loss_train.set_xlabel("Epoch", color="red")
        self.ax_loss_train.set_ylabel("Loss", color="red")

        self.ax_loss_val.set_title("Validation Loss", color="red")
        self.ax_loss_val.set_xlabel("Epoch", color="red")
        self.ax_loss_val.set_ylabel("Loss", color="red")

        self.ax_acc_train.set_title("Train Accuracy", color="red")
        self.ax_acc_train.set_xlabel("Epoch", color="red")
        self.ax_acc_train.set_ylabel("Accuracy", color="red")

        self.ax_acc_val.set_title("Validation Accuracy", color="red")
        self.ax_acc_val.set_xlabel("Epoch", color="red")
        self.ax_acc_val.set_ylabel("Accuracy", color="red")

    def on_train_begin(self, logs=None):
        self.epochs.clear()
        self.tamper_loss.clear()
        self.val_tamper_loss.clear()
        self.tamper_acc.clear()
        self.val_tamper_acc.clear()

    def plot_series_colors(self, ax, x_vals, y_vals):

        if len(x_vals) < 2:
            return

        for i in range(1, len(x_vals)):
            x0, x1 = x_vals[i-1], x_vals[i]
            y0, y1 = y_vals[i-1], y_vals[i]

            if y0 is None or y1 is None:
                continue

            color = "green" if y1> y0 else "red"
            ax.plot([x0, x1], [y0, y1], color=color)

        last_y = y_vals[-1]
        if last_y is not None:
            ax.plot(x_vals[-1], last_y, marker="o", color="red", markersize=4)

    def style_axis(self, ax, title: str, y_label: str):
        ax.clear()
        ax.set_facecolor("black")
        ax.set_title(title, color="red")
        ax.set_xlabel("Epoch", color="red")
        ax.set_ylabel(y_label, color="red")
        ax.tick_params(colors="red")
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch+1)

        self.tamper_loss.append(logs.get("tamper_output_loss", None))
        self.val_tamper_loss.append(logs.get("val_tamper_output_loss", None))
        self.tamper_acc.append(logs.get("tamper_output_accuracy", None))
        self.val_tamper_acc.append(logs.get("val_tamper_output_accuracy", None))

        # Train loss axis  # NEW
        self.style_axis(self.ax_loss_train, "Train Loss", "Loss")  # NEW
        self.plot_series_colors(self.ax_loss_train, self.epochs, self.tamper_loss)  # NEW

        # Validation loss axis  # NEW
        self.style_axis(self.ax_loss_val, "Validation Loss", "Loss")  # NEW
        self.plot_series_colors(self.ax_loss_val, self.epochs, self.val_tamper_loss)  # NEW

        # Train accuracy axis  # NEW
        self.style_axis(self.ax_acc_train, "Train Accuracy", "Accuracy")  # NEW
        self.plot_series_colors(self.ax_acc_train, self.epochs, self.tamper_acc)  # NEW

        # Validation accuracy axis  # NEW
        self.style_axis(self.ax_acc_val, "Validation Accuracy", "Accuracy")  # NEW
        self.plot_series_colors(self.ax_acc_val, self.epochs, self.val_tamper_acc)  # NEW

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()




def build_dataset():
    original_images = load_original_images()
    if not original_images:
        raise RuntimeError("No images found")

    resized_originals = [cv2.resize(img, IMG_SIZE) for img in original_images]

    x = []
    y_tamper = []
    y_type = []

    type_to_index = {name: idx for idx, name in enumerate(TYPE_LABELS)}

    for base_img_bgr in resized_originals:
        x.append(preprocess_for_model(base_img_bgr))
        y_tamper.append(0)
        y_type.append(type_to_index["original"])

        tamper_generators = [(tamper_jpeg, "jpeg"), (tamper_blur, "blur"), (tamper_noise, "noise"), (tamper_copy_move, "copy_move")]

        for tamper_func, tname in tamper_generators:
            tampered_bgr = tamper_func(base_img_bgr)
            x.append(preprocess_for_model(tampered_bgr))
            y_tamper.append(1)
            y_type.append(type_to_index[tname])

        if len(resized_originals) > 1:
            donor_candidates = [img for img in resized_originals if img is not base_img_bgr]
            donor_img = random.choice(donor_candidates)
        else:
            donor_img = base_img_bgr

        spliced_bgr = tamper_splice(base_img_bgr, donor_img)
        x.append(preprocess_for_model(spliced_bgr))
        y_tamper.append(1)
        y_type.append(type_to_index["splice"])

        inpainted_bgr = tamper_inpaint(base_img_bgr)
        x.append(preprocess_for_model(inpainted_bgr))
        y_tamper.append(1)
        y_type.append(type_to_index["inpaint"])

    x = np.stack(x, axis=0)
    y_tamper = np.array(y_tamper, dtype="int32")
    y_type = np.array(y_type, dtype="int32")

    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(.8 * num_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train, x_val = x[train_idx], x[val_idx]
    y_tamper_train, y_tamper_val = y_tamper[train_idx], y_tamper[val_idx]
    y_type_train, y_type_val = y_type[train_idx], y_type[val_idx]

    return x_train, x_val, y_tamper_train, y_tamper_val, y_type_train, y_type_val

def save_class_indices():
    data = {"tamper": {i: name for i, name in enumerate(TAMPER_LABELS)},
            "type": {i: name for i, name in enumerate(TYPE_LABELS)},}
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    ensure_dirs()
    print(f"Loading original images from: {ORIGINAL_DIR}")
    (x_train, x_val, y_tamper_train, y_tamper_val, y_type_train, y_type_val) = build_dataset()

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")

    model = build_tamper_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                                   num_tamper_classes=len(TAMPER_LABELS),
                                   num_type_classes = len(TYPE_LABELS),
                                   )
    model.compile(optimizer="adam", loss= {
        "tamper_output": "sparse_categorical_crossentropy",
        "type_output": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "tamper_output": 1.0,
        "type_output": 0.5,
    },
    metrics={
        "tamper_output": "accuracy",
        "type_output": "accuracy",
    },

    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir= str(LOG_DIR),
        histogram_freq=1,
        update_freq="epoch",
    )

    live_plot_cb = LiveTrainingPlot()

    callbacks = [live_plot_cb,
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_tamper_output_loss", factor=.5, patience=3, verbose = 1, mode= "min"),
                 tf.keras.callbacks.EarlyStopping(monitor="val_tamper_output_loss", patience=5, restore_best_weights=True, verbose = 1, mode= "min",),
                ]

    history = model.fit(x_train, {"tamper_output": y_tamper_train, "type_output": y_type_train},
                        validation_data=(x_val, {"tamper_output": y_tamper_val, "type_output": y_type_val},
                        ),
                        epochs=15,
                        batch_size=16,
                        callbacks=callbacks,
                        verbose=1,
                        )

    print("Training complete. Saving model...")
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    print("Saving class index mapping...")
    save_class_indices()
    print(f"Class index mapping saved to {CLASS_MAP_PATH}")


if __name__ == "__main__":
    main()

