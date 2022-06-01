from pickletools import optimize
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px


def generate_data(DATA_POINTS=60, COLUMNS=3, SEED=42):
    X = np.zeros((DATA_POINTS, COLUMNS))
    np.random.seed(SEED)
    angles = (np.random.rand(DATA_POINTS) ** 3 + 0.5) * 2 * np.pi # uneven destribution
    X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) + 0.5 # oval shape
    X += 0.28 * np.random.randn(DATA_POINTS, COLUMNS)  # adding more noise
    return X + [0.3, 0, 0.3]

def get_prediction(X_train, SEED=42, LR=0.5, EPOCHS=500):
    tf.random.set_seed(SEED)

    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(2)
    ])
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(3)
    ])

    autoencoder = tf.keras.Sequential([encoder, decoder])

    optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
    autoencoder.compile(loss="mse", optimizer=optimizer)

    history = autoencoder.fit(X_train, X_train, epochs=EPOCHS)
    return encoder.predict(X_train)


if __name__ == '__main__':
    X_train = generate_data()
    df = pd.DataFrame(X_train, columns=['x', 'y', 'z'])
    print(f"description of df:\n{df.describe().T}")

    fig = px.scatter_3d(df, x='x', y='y', z='z', opacity=0.7)
    fig.show()

    encodings = get_prediction(X_train)
    pred_df = pd.DataFrame(np.c_[encodings, np.zeros(encodings.shape[0])], columns=["new_x", "new_y", "None"])

    print(f"description of pred_df:\n{pred_df.describe().T}")

    fig = px.scatter_3d(pred_df, x='new_x', y='new_y', z='None', opacity=0.7)
    fig.show()

