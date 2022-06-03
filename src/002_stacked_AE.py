import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_data(normalize_by=255, N=5000):
    """gets the MNIST dataset

    Args:
        normalize_by (int, optional): normalization value. Defaults to 255.
        N (int, optional): no. datapoints in validation set_seed. Defaults to 5000.

    Returns:
        tuple: tuple of train,valid,test data
    """
    mnist = tf.keras.datasets.mnist.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = mnist
    X_train_full = X_train_full.astype(np.float32) / normalize_by
    X_test = X_test.astype(np.float32) / normalize_by

    X_train, X_valid = X_train_full[:-N], X_train_full[-N:]
    print(f"shape of- \nX_train: {X_train.shape}, \nX_valid, {X_valid.shape}")
    y_train, y_valid = y_train_full[:-N], y_train_full[-N:]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_model(SEED=42):

    tf.random.set_seed(SEED)

    LAYERS_encoder = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(30, activation="relu")
    ]
    stacked_encoder = tf.keras.Sequential(LAYERS_encoder)

    LAYERS_decoder = [
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(28*28),
        tf.keras.layers.Reshape([28,28])
    ]

    stacked_decoder = tf.keras.Sequential(LAYERS_decoder)

    stacked_autoencoder = tf.keras.Sequential([
        stacked_encoder,
        stacked_decoder
    ])

    return stacked_autoencoder

def plot_results():

    pass


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data()