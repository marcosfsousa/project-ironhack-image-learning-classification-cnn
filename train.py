
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from config import LEARNING_RATE


    
# train.py

def compile_model(model):
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=LEARNING_RATE, 
        momentum=0.9,
        nesterov=True
    )
    model.compile(
        optimizer=optimizer, #changed from adam to Adam(learning_rate=1e-3) for better control over learning rate should start at 0.001
        loss=sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )
    return model
                           #changed from Adam to SGD (momentum + Nesterov) adapts the learning rate
                           # each parameter individually, allowing the model to 
                           #converge faster and achieve better early accuracy than
                           #standard SGD, especially on datasets like CIFAR-10 with small CNNs.
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5, # changed from 1e-6 to 1e-5 to prevent the learning rate from becoming too small
    verbose=1,
)
                           # reduce learning rate when a metric has stopped improving
                           #  can help the model slow down and fine tune.
    
def train_model(model, x_train, y_train, batch_size, epochs):
    early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[lr_scheduler, early_stop],
        verbose=1,
    )
    return history


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy