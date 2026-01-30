# metrics.py
import numpy as np

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def evaluate_model(model, test_ds):
    return model.evaluate(test_ds, verbose=0)

def merge_histories(histories):
    merged = {}
    for h in histories.values():
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged

def predict_classes(model, test_ds):
    """
    Predict class indices for a tf.data.Dataset.
    """
    probs = model.predict(test_ds, verbose=0)
    return np.argmax(probs, axis=1)