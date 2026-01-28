# model.py
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from config import INPUT_SHAPE, NUM_CLASSES


def build_model():
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model
