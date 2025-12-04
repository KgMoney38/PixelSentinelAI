#Kody Graham
#12/01/2025
#This class will define my TensorFlow model

import tensorflow as tf
from tensorflow.keras import layers, models

def build_tamper_model(
        input_shape = (224,224,3),
        num_tamper_classes: int = 2,
        num_type_classes: int = 7,
) -> tf.keras.Model:

    #Build simple CNN with 2 heads

    inputs = layers.Input(shape=input_shape)

    #Backbone CNN
    x = layers.Conv2D(32, (3, 3), activation = "relu", padding="same")(inputs)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, (3, 3), activation = "relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, (3, 3), activation = "relu", padding="same", name= "last_conv")(x)
    x = layers.MaxPool2D()(x)

    last_conv_output = x

    x = layers.Flatten()(last_conv_output)
    x= layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    tamper_output = layers.Dense(num_tamper_classes, activation="softmax", name= "tamper_output")(x)
    type_output = layers.Dense(num_type_classes, activation="softmax", name= "type_output")(x)

    model = models.Model(inputs=inputs, outputs=[tamper_output, type_output], name="tamper_detector")

    return model