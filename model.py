# model.py
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import (
        Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten,
        GlobalAveragePooling2D, BatchNormalization, Resizing
    )

def build_scratch_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    return model

def set_backbone_trainable(model, trainable: bool):
    model.get_layer("backbone").trainable = trainable

def build_transfer_model(input_shape, num_classes, backbone_trainable=False):
    backbone = VGG16(
        name="backbone",
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    backbone.trainable = backbone_trainable

    model = Sequential([
        Input(shape=input_shape),
        Resizing(64, 64),
        backbone,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model


def build_model(strategy: str, **kwargs):
    if strategy == "scratch":
        return build_scratch_model(**kwargs)

    if strategy == "transfer":
        return build_transfer_model(**kwargs)

    raise ValueError(f"Unknown strategy: {strategy}")
