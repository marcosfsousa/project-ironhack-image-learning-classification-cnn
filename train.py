# train.py
from keras.optimizers import Adam

def compile_model(model, lr):
    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="SparseCategoricalCrossentropy",
        metrics=["accuracy"]
    )

def train_model(
    model,
    train_data,
    val_data,
    compile_fn,
    phases
):
    histories = {}

    for phase in phases:
        if "set_trainable" in phase:
            phase["set_trainable"](model)

        compile_fn(
            model,
            lr=phase["learning_rate"])

        callbacks = phase.get("base_callbacks", []) + phase.get("callbacks", [])

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=phase["epochs"],
            callbacks=callbacks
        )

        histories[phase["name"]] = history

    return histories
