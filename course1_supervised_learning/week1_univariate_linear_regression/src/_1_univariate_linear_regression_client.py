import matplotlib.pyplot as plt
import numpy as np
from _1_plotters_for_visualizations import (
    plot_loss_curves_for_lrs,
    plot_loss_curves_for_lrs_with_subplot,
    plot_model_fit,
    plot_training_progress,
)
from _1_univariate_linear_regression import gradient_descent_with_loss_gradient_tol

# Generate synthetic data
np.random.seed(42)
n = 100
w_true, b_true = 2.5, -1.0
noise_std = 0.5

X_train = np.random.uniform(-10, 10, size=n)
noise = np.random.normal(0, noise_std, size=n)
Y_train = w_true * X_train + b_true + noise

learning_rates = [0.005, 0.008, 0.01, 0.03, 0.05]
all_histories = {}  # Collect results to optionally inspect later

for lr in learning_rates:
    diverged = False
    loss_history = []
    grad_norm_history = []
    w_history = []
    b_history = []
    try:
        # Fit model using loss/grad_tol based gradient descent
        loss_history, grad_norm_history, w_history, b_history = (
            gradient_descent_with_loss_gradient_tol(X_train, Y_train, learning_rate=lr)
        )
        if np.isnan(loss_history[-1]) or np.isinf(loss_history[-1]):
            raise ValueError
    except:
        print(f"Learning rate {lr} caused divergence.")
        diverged = True

    all_histories[lr] = {
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "w_history": w_history,
        "b_history": b_history,
        "Diverged": diverged,
    }

# Get the final loss for all learning rates that did not diverge
loss_by_lr = {
    lr: history["loss_history"]
    for lr, history in all_histories.items()
    if not history["Diverged"]
}

# Plot the loss curve for all learning rates
plot_loss_curves_for_lrs_with_subplot(loss_by_lr)

print("Final losses by learning rate:")
for lr, losses in loss_by_lr.items():
    print(f"LR={lr}: Final Loss = {losses[-1]:.6f}")

# Select best learning rate based on final loss
# Select best learning rate based on final loss
best_lr = min(
    loss_by_lr.items(),
    key=lambda item: item[1][
        -1
    ],  # item is (lr, loss_history), so item[1][-1] is final loss
)[0]

print(f"Best learning rate based on final loss: {best_lr}")


loss_history = all_histories[best_lr]["loss_history"]
grad_norm_history = all_histories[best_lr]["grad_norm_history"]
w_history = all_histories[best_lr]["w_history"]
b_history = all_histories[best_lr]["b_history"]

print(
    f"Final w : {w_history[-1]}, Final b : {b_history[-1]}, loss : {loss_history[-1]}"
)

# Plot the model fit for the best learning_rate
plot_model_fit(X_train, Y_train, w_true, b_true, w_history[-1], b_history[-1])

# Plot the training progress for the best learning rate
plot_training_progress(loss_history, grad_norm_history, w_history, b_history)


# Fit model using patience based gradient descent
# val_X, val_Y = X_train[:20], Y_train[:20]
# train_X, train_Y = X_train[20:], Y_train[20:]
# w_pred_2, b_pred_2, loss_history_2 = gradient_descent_patience_based(
#     train_X, train_Y, val_X=val_X, val_Y=val_Y, val_patience=10
# )
