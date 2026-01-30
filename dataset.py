import config 
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split


def load_cifar10():
    """
    Load CIFAR-10 dataset from Keras.

    Returns:
        (x_train, y_train), (x_test, y_test)
        - x_* shape: (N, 32, 32, 3), dtype uint8
        - y_* shape: (N, 1)
    """
    return cifar10.load_data()


def preprocess_cifar10(x, y):
    """
    Preprocess CIFAR-10 data:
    - Normalize images to [0, 1]
    - Flatten labels to shape (N,)

    Args:
        x: numpy array of images
        y: numpy array of labels

    Returns:
        x_processed, y_processed
    """
    x = x.astype("float32") / 255.0
    y = y.squeeze()

    return x, y


def get_cifar10_data():
    """
    Full CIFAR-10 pipeline:
    load → preprocess

    Returns:
        x_train, y_train, x_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    x_train, y_train = preprocess_cifar10(x_train, y_train)
    x_test, y_test = preprocess_cifar10(x_test, y_test)

    return x_train, y_train, x_test, y_test


def get_cifar10_datasets(batch_size, val_split=0.1):
    x_train, y_train, x_test, y_test = get_cifar10_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_split, random_state=config.RANDOM_SEED
    )

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

