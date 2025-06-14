import matplotlib.pyplot as plt
import numpy as np


def plot_model_fit(
    X,
    Y,
    w_true,
    b_true,
    w_pred,
    b_pred,
    title="Model Fit",
    label_pred="Model Fit Tol based",
):
    """
    Plot model prediction vs true line and noisy data.

    Parameters:
    - X, Y: Training data
    - w_true, b_true: True line parameters
    - w_pred, b_pred: Predicted model parameters
    - title: Title of the plot
    - label_pred: Label for the predicted model line
    """
    sorted_X = np.sort(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.7, label="Noisy Data")
    plt.plot(sorted_X, w_true * sorted_X + b_true, "r-", label="True Line")
    plt.plot(sorted_X, w_pred * sorted_X + b_pred, "g--", label=label_pred)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss_curves_for_lrs(loss_by_lr):
    """
    Plots loss vs iterations for multiple learning rates.

    Parameters:
    -----------
    loss_by_lr : dict
        A dictionary where keys are learning rates (float)
        and values are lists of loss values over iterations.
    """
    plt.figure(figsize=(12, 6))
    for lr, losses in loss_by_lr.items():
        plt.plot(losses, label=f"LR={lr} (Loss={losses[-1]:.6f})")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison Across Learning Rates")
    plt.legend(title="Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss_curves_for_lrs_with_subplot(loss_by_lr, zoom_range=20):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs[0].set_title("Loss Curves (Full)")
    axs[1].set_title(f"Loss Curves (Zoomed to Iter 0–{zoom_range})")

    for lr, losses in loss_by_lr.items():
        final_loss = losses[-1]
        label = f"LR={lr} (Loss={final_loss:.6f})"
        axs[0].plot(losses, label=label)
        axs[1].plot(
            range(min(zoom_range, len(losses))), losses[:zoom_range], label=label
        )

    for ax in axs:
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_metric(ax, x_vals, y_vals, xlabel, ylabel, color):

    # Plot the line
    title = f"{ylabel} over {xlabel}"
    ax.plot(x_vals, y_vals, label=title, color=color)

    # Final point coordinates
    final_x = x_vals[-1]
    final_y = y_vals[-1]

    # Plot final point
    ax.scatter(final_x, final_y, color="blue", s=30, zorder=5, label="Final Value")

    # Dotted reference lines
    ax.axvline(final_x, linestyle="--", color="gray", linewidth=1)
    ax.axhline(final_y, linestyle="--", color="gray", linewidth=1)

    # Annotate the final value
    ax.annotate(
        f"({final_x},{final_y:.5f})",
        xy=(x_vals[-1], y_vals[-1]),
        xytext=(-40, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=12,
    )

    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


def plot_training_progress(loss, grad_norm, w, b):
    fig, axs = plt.subplots(2, 2, figsize=(20, 9))
    x_vals = list(range(len(w)))

    plot_metric(axs[0, 0], x_vals, loss, "Iteration", "Loss (MSE)", "blue")
    plot_metric(axs[0, 1], x_vals, grad_norm, "Iteration", "Gradient Norm", "red")
    plot_metric(axs[1, 0], x_vals, w, "Iteration", "w (weight)", "orange")
    plot_metric(axs[1, 1], x_vals, b, "Iteration", "b (bias)", "green")

    plt.tight_layout()
    plt.show()
