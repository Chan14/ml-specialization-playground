import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


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


def prepare_dataset(filename):
    data = np.loadtxt(filename, delimiter=",")
    x, y = data[:, :-1], data[:, -1]
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=80)
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_, y_, test_size=0.50, random_state=80
    )
    return x_train, y_train, x_cv, y_cv, x_test, y_test


def train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=None):
    train_mses = []
    cv_mses = []
    degrees = range(1, max_degree + 1)

    for degree in degrees:
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

        model.fit(X_train_mapped_scaled, y_train)

        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        X_cv_mapped = poly.transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    degrees = [str(d) for d in degrees]
    plt.plot(degrees, train_mses, marker="o", c="r", label="training MSEs")
    plt.plot(degrees, cv_mses, marker="o", c="b", label="CV MSEs")
    plt.plot(
        degrees, np.repeat(baseline, len(degrees)), linestyle="--", label="baseline"
    )

    plt.title("degree of polynomial vs. train and CV MSEs")
    plt.xticks(degrees)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_plot_reg_params(
    reg_params, x_train, y_train, x_cv, y_cv, degree=1, baseline=None
):
    train_mses = []
    cv_mses = []
    models = []
    scalers = []

    for reg_param in reg_params:
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        model = Ridge(alpha=reg_param)
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        X_cv_mapped = poly.transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    reg_params = [str(x) for x in reg_params]
    plt.plot(reg_params, train_mses, marker="o", c="r", label="training MSEs")
    plt.plot(reg_params, cv_mses, marker="o", c="b", label="CV MSEs")
    plt.plot(
        reg_params,
        np.repeat(baseline, len(reg_params)),
        linestyle="--",
        label="baseline",
    )
    plt.title("lambda vs. train and CV MSEs")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_plot_diff_datasets(model, files, max_degree, baseline=None):

    for f in files:

        x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset(f["filename"])

        train_mses = []
        cv_mses = []
        degrees = range(1, max_degree + 1)
        degrees_str = [str(d) for d in degrees]

        for degree in degrees:
            poly = PolynomialFeatures(degree, include_bias=False)
            X_train_mapped = poly.fit_transform(x_train)

            scaler_poly = StandardScaler()
            X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

            model.fit(X_train_mapped_scaled, y_train)

            yhat = model.predict(X_train_mapped_scaled)
            train_mse = mean_squared_error(y_train, yhat) / 2
            train_mses.append(train_mse)

            X_cv_mapped = poly.transform(x_cv)
            X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
            yhat = model.predict(X_cv_mapped_scaled)
            cv_mse = mean_squared_error(y_cv, yhat) / 2
            cv_mses.append(cv_mse)

        plt.plot(
            degrees_str,
            train_mses,
            marker="o",
            c="r",
            linestyle=f["linestyle"],
            label=f"{f['label']} training MSEs",
        )
        plt.plot(
            degrees_str,
            cv_mses,
            marker="o",
            c="b",
            linestyle=f["linestyle"],
            label=f"{f['label']} CV MSEs",
        )
    plt.plot(
        degrees_str,
        np.repeat(baseline, len(degrees_str)),
        linestyle="--",
        label="baseline",
    )

    plt.title("degree of polynomial vs. train and CV MSEs")
    plt.xticks(degrees_str)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_plot_learning_curve(
    model, x_train, y_train, x_cv, y_cv, degree=1, baseline=None
):

    # 1. Shuffle the entire training and CV sets ONCE at the start
    # This ensures that [:num_samples] gives us a random, representative slice.
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_cv, y_cv = shuffle(x_cv, y_cv, random_state=42)

    train_mses = []
    cv_mses = []
    num_samples_train_and_cv = []
    percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for percent in percents:
        num_samples_train = round(len(x_train) * (percent / 100.0))
        num_samples_cv = round(len(x_cv) * (percent / 100.0))
        num_samples_train_and_cv.append(num_samples_train + num_samples_cv)

        x_train_sub, y_train_sub = (
            x_train[:num_samples_train],
            y_train[:num_samples_train],
        )
        x_cv_sub, y_cv_sub = x_cv[:num_samples_cv], y_cv[:num_samples_cv]

        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train_sub)

        # Scale the features
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

        # Fit the model
        model.fit(X_train_mapped_scaled, y_train_sub)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train_sub, yhat) / 2
        train_mses.append(train_mse)

        # Compute the CV MSE
        X_cv_mapped = poly.transform(x_cv_sub)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv_sub, yhat) / 2
        cv_mses.append(cv_mse)

    # Plot the results
    plt.plot(
        num_samples_train_and_cv, train_mses, marker="o", c="r", label="training MSEs"
    )
    plt.plot(num_samples_train_and_cv, cv_mses, marker="o", c="b", label="CV MSEs")
    plt.plot(
        num_samples_train_and_cv,
        np.repeat(baseline, len(percents)),
        linestyle="--",
        label="baseline",
    )
    plt.title("number of examples vs. train and CV MSEs")
    plt.xlabel("total number of training and cv examples")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
