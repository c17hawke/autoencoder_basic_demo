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

def get_model():

    pass

def plot_results():

    pass


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data()