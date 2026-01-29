# model.py
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import (
        Input, Dense, Dropout,
        GlobalAveragePooling2D, BatchNormalization, Resizing
    )
from config import INPUT_SHAPE, NUM_CLASSES

def build_model():
    backbone = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    backbone.trainable = True  # freeze backbone

    model = Sequential([
        Input(shape=INPUT_SHAPE),
        Resizing(64, 64),  # upscaler
        
        backbone,          # VGG16 feature extractor

        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    return model
