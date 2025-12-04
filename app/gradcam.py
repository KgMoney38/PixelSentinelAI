#Kody Graham
#12/02/2025
#This class will generate the heatmap overlay.

from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "last_conv",
    pred_index: Optional[int] = None,
    ) -> np.ndarray:

    last_conv_layer = model.get_layer(last_conv_layer_name)
    tamper_output = model.get_layer("tamper_output").output

    grad_model = tf.keras.models.Model([model.input], [last_conv_layer.output, tamper_output],)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:,pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap= tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(
        original_img_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = .4,
) -> np.ndarray:

    h,w = original_img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255* heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, original_img_bgr, 1 - alpha, 0)

    return overlay