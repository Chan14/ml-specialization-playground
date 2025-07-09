import logging
import math
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def plot_scatterplots(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    features: List[str],
    target: str,
    output_path: Optional[Union[str, Path]] = None,
    W: Optional[np.ndarray] = None,
    b: Optional[float] = None,
    X_scaled: Optional[np.ndarray] = None,
    n_cols: int = 3,  # number of columns
    share_y_axis: bool = True,  # New parameter for Y-axis sharing
):
    """
    Plots scatter plots for features from a NumPy array against a NumPy array target and optional NumPy array predictions.

    Args:
        X_train (np.ndarray): The input feature set (2D NumPy array).
                                     Columns are accessed by index corresponding to 'features' list order.
        Y_train (np.ndarray): The target values (2D NumPy array).
        features (List[str]): A list of feature names (strings) used for plot titles and labels.
                              The order must match the column order in input_features.
        target (str): The name (string) of the target
        output_path (Path or str or None): Optional full path including filename where the plot will be saved.
                                          Can be a string or a pathlib.Path object.
        W (np.ndarray or None): Optional weights (2D NumPy array) for predictions, or None.
        b (float or None): Optional model bias (scalar) for predictions, or None.
        X_scaled (np.ndarray or None): The scaled input feature set (2D NumPy array) if the feature set was scaled,
                                        None otherwise. Used for predictions if provided.
        n_cols (int): Number of columns to use in the subplot grid. Defaults to 3.
        share_y_axis (bool): If True, all scatter plots will share the same y-axis range. Defaults to True.
    """

    if X_train.shape[1] != len(features):
        raise ValueError(
            "Mismatch between number of columns in X_train and number of feature names."
        )
    if Y_train.ndim != 2:
        raise ValueError("Y_train must be a 2D array.")

    Y_pred = None
    if W is not None and b is not None:
        if not isinstance(W, np.ndarray) or W.ndim != 2:
            raise TypeError("W must be a 2D NumPy array.")
        if not isinstance(b, (int, float)):
            raise TypeError("b must be a scalar (int or float).")

        X_for_prediction = X_scaled if X_scaled is not None else X_train
        if X_for_prediction.shape[1] != W.shape[0]:
            raise ValueError(
                "Mismatch between number of features in X_for_prediction and number of weights in W."
            )
        Y_pred = np.dot(X_for_prediction, W) + b
    elif W is not None or b is not None:
        logging.warning(
            "Both 'W' and 'b' must be provided to plot predictions. Skipping predictions."
        )

    m, n = X_train.shape

    cols = n_cols
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 6, rows * 6), squeeze=False
    )  # squeeze=False for consistent 2D array
    fig.suptitle(
        f"Feature vs Target Relationships (Scatter Plots)",
        fontsize=16,
        fontweight="bold",
    )
    axes = axes.flatten()

    logging.info("Generating scatter plots...")

    # Initialize the first scatter plot's y-axis for sharing if enabled
    first_scatter_ax = axes[0]

    for i, feature in enumerate(features):
        ax = axes[i]
        if share_y_axis and i > 0:
            ax.sharey(first_scatter_ax)

        ax.scatter(
            X_train[:, i], Y_train, alpha=0.7, color="lightcoral", label="Actual Target"
        )
        if Y_pred is not None:
            ax.scatter(
                X_train[:, i],
                Y_pred,
                alpha=0.7,
                color="skyblue",
                label="Model Prediction",
            )

        ax.set_title(f"{feature} vs {target}")
        ax.set_xlabel(feature)

        # Only set y-label for the first plot in each row or if y-axis is not shared
        if i % cols == 0 or not share_y_axis:
            ax.set_ylabel(target)

        # Place legend on each subplot if predictions are present
        if Y_pred is not None:
            ax.legend()

    # Hide any unused subplots
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to prevent suptitle overlap

    if output_path:
        output_path = Path(output_path)  # Ensure it's a Path object
        plt.savefig(output_path, dpi=150)
        logging.info(f"Scatter plots saved to '{output_path}'")

    plt.show()


def plot_histograms(
    X_train: np.ndarray,
    features: List[str],
    n_cols: int = 3,  # number of columns
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Plots histograms for features from a NumPy array.

    Args:
        X_train (np.ndarray): The input feature set (2D NumPy array).
                                     Columns are accessed by index corresponding to 'features' list order.
        features (List[str]): A list of feature names (strings) used for plot titles and labels.
                              The order must match the column order in input_features.
        n_cols (int): Number of columns to use in the subplot grid. Defaults to 3.
        output_path (Path or str or None): Optional full path including filename where the plot will be saved.
                                          Can be a string or a pathlib.Path object.
    """
    m, n = X_train.shape

    cols = n_cols
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 6, rows * 6), squeeze=False
    )  # squeeze=False for consistent 2D array
    fig.suptitle("Feature Distributions (histograms)", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    logging.info("Generating histograms...")
    for i, feature in enumerate(features):
        ax = axes[i]
        if X_train.shape[1] <= i:
            logging.warning(
                f"Feature '{feature}' (index {i}) is out of bounds for input_features array. Skipping."
            )
            fig.delaxes(ax)  # Remove unused subplot
            continue
        ax.hist(X_train[:, i], color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
    # Hide any unused plots
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to prevent suptitle overlap

    if output_path:
        output_path = Path(output_path)  # Ensure it's a Path object
        plt.savefig(output_path, dpi=150)
        logging.info(f"histograms saved to '{output_path}'")

    plt.show()
