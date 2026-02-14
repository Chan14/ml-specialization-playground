import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def plot_dataset(x, y, title):
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["lines.markersize"] = 12
    plt.scatter(x, y, marker="x", c="r")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title):
    plt.scatter(x_train, y_train, marker="x", c="r", label="training")
    plt.scatter(x_cv, y_cv, marker="o", c="b", label="cross validation")
    plt.scatter(x_test, y_test, marker="^", c="g", label="test")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def plot_train_cv_mses(degree, train_mses, cv_mses, title):
    plt.plot(degree, train_mses, marker="o", c="r", label="Training MSEs")
    plt.plot(degree, cv_mses, marker="o", c="b", label="CV MSEs")
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def plot_bc_dataset(x, y, title):
    pos = y.squeeze() == 1
    plt.scatter(x[pos, 0], x[pos, 1], marker="x", c="r", label="y=1")
    plt.scatter(x[~pos, 0], x[~pos, 1], marker="o", c="b", label="y=0")
    plt.title("x1 vs x2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_nn_loss_curves(histories, alpha, seeds):
    # Create a figure with 5 subplots (one for each seed)
    fig, axes = plt.subplots(2, 3, figsize=(25, 14), sharey=True)
    fig.suptitle(f"Training Loss Curves per Seed (LR={alpha})", fontsize=16)

    model_names = ["model_1", "model_2", "model_3"]
    for idx, seed in enumerate(seeds):
        ax = axes.flat[idx]

        # histories[seed] contains [hist_m1, hist_m2, hist_m3]
        for m_idx, history in enumerate(histories[seed]):
            ax.plot(history.history["loss"], label=model_names[m_idx])

        ax.set_title(f"Seed: {seed}")
        ax.set_xlabel("Epochs")
        if idx == 0:
            ax.set_ylabel("MSE (Loss)")
        ax.set_yscale("log")  # Log scale makes it easier to see convergence
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def build_models(random_state=None):

    if random_state:
        tf.random.set_seed(random_state)

    model_1 = Sequential(
        [
            Dense(25, activation="relu", kernel_initializer="he_normal"),
            Dense(15, activation="relu", kernel_initializer="he_normal"),
            Dense(1, activation="linear", kernel_initializer="glorot_uniform"),
        ],
        name="model_1",
    )

    model_2 = Sequential(
        [
            Dense(20, activation="relu", kernel_initializer="he_normal"),
            Dense(12, activation="relu", kernel_initializer="he_normal"),
            Dense(12, activation="relu", kernel_initializer="he_normal"),
            Dense(20, activation="relu", kernel_initializer="he_normal"),
            Dense(1, activation="linear", kernel_initializer="glorot_uniform"),
        ],
        name="model_2",
    )

    model_3 = Sequential(
        [
            Dense(32, activation="relu", kernel_initializer="he_normal"),
            Dense(16, activation="relu", kernel_initializer="he_normal"),
            Dense(8, activation="relu", kernel_initializer="he_normal"),
            Dense(4, activation="relu", kernel_initializer="he_normal"),
            Dense(12, activation="relu", kernel_initializer="he_normal"),
            Dense(1, activation="linear", kernel_initializer="glorot_uniform"),
        ],
        name="model_3",
    )

    model_list = [model_1, model_2, model_3]

    return model_list


def build_models2(random_state=None):

    if random_state:
        tf.random.set_seed(random_state)

    model_1 = Sequential(
        [
            Dense(25, activation="relu"),
            Dense(15, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="model_1",
    )

    model_2 = Sequential(
        [
            Dense(20, activation="relu"),
            Dense(12, activation="relu"),
            Dense(12, activation="relu"),
            Dense(20, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="model_2",
    )

    model_3 = Sequential(
        [
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="relu"),
            Dense(12, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="model_3",
    )

    model_list = [model_1, model_2, model_3]

    return model_list
