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

def get_model(X_train):
    pass

if __name__ == '__main__':
    X_train = generate_data()
    df = pd.DataFrame(X_train, columns=['x', 'y', 'z'])
    print(f"description of df:\n{df.describe().T}")

    fig = px.scatter_3d(df, x='x', y='y', z='z', opacity=0.7)
    fig.show()

    enocdings = get_model(X_train)
