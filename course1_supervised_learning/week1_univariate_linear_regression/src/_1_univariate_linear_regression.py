import numpy as np


def compute_loss(X, Y, w, b):
    """
    Compute the Mean Squared Error (MSE) for univariate linear regression.
    """
    Y_pred = w * X + b
    return np.mean((Y_pred - Y) ** 2)


def compute_gradients(X, Y, w, b):
    """
    Compute gradients of MSE with respect to weight and bias.

    Returns:
        (dw, db): Tuple of gradients.
    """
    Y_pred = w * X + b
    error = Y_pred - Y
    dw = np.mean(error * X)
    db = np.mean(error)
    return dw, db


def gradient_descent_with_loss_gradient_tol(
    X,
    Y,
    w_init=0.0,
    b_init=0.0,
    learning_rate=0.01,
    max_iters=1000,
    loss_tol=1e-6,
    grad_tol=1e-6,
    verbose=True,
    print_every=100,
    print_first=5,
    print_last=5,
):
    """
    Gradient descent with early stopping based on loss change or gradient norm.

    Returns:
        w, b, loss_history: Final weight, bias, and list of losses.
    """
    assert X.shape == Y.shape, "X and Y must be the same shape"
    w, b = w_init, b_init
    prev_loss = None
    # For plotting

    w_history = []
    b_history = []
    grad_norm_history = []
    loss_history = []

    for i in range(max_iters):
        loss = compute_loss(X, Y, w, b)
        loss_history.append(loss)
        dw, db = compute_gradients(X, Y, w, b)
        grad_norm = np.sqrt(dw**2 + db**2)
        grad_norm_history.append(grad_norm)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        w_history.append(w)
        b_history.append(b)

        # Logging logic
        should_print = (
            i < print_first or i % print_every == 0 or i >= max_iters - print_last
        )
        if verbose and should_print:
            print(
                f"Iter {i}: Loss = {loss:.5f}, w = {w:.5f}, b = {b:.5f}, grad_norm = {grad_norm:.5f}"
            )

        # Early stopping
        if prev_loss is not None and abs(loss - prev_loss) < loss_tol:
            if verbose:
                print(
                    f"Early stopping at iter {i} — Δloss = {abs(loss - prev_loss):.2e} < {loss_tol}"
                )
            break
        if grad_norm < grad_tol:
            if verbose:
                print(
                    f"Early stopping at iter {i} — grad_norm = {grad_norm:.2e} < {grad_tol}"
                )
            break

        prev_loss = loss
    print(f"Final w : {w}, Final b : {b}, loss : {loss}, grad_norm : {grad_norm}")

    return (loss_history, grad_norm_history, w_history, b_history)


def gradient_descent_patience_based(
    X,
    Y,
    w_init=0.0,
    b_init=0.0,
    learning_rate=0.01,
    max_iters=1000,
    verbose=True,
    print_every=100,
    print_first=5,
    print_last=5,
    val_X=None,
    val_Y=None,
    val_patience=10,
    min_delta=1e-6,
):
    """
    Gradient descent with early stopping based on validation loss patience.

    Returns:
        w, b, loss_history: Final weight, bias, and list of training losses.
    """
    assert X.shape == Y.shape, "X and Y must be the same shape"
    w, b = w_init, b_init
    loss_history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for i in range(max_iters):
        loss = compute_loss(X, Y, w, b)
        loss_history.append(loss)
        dw, db = compute_gradients(X, Y, w, b)
        grad_norm = np.sqrt(dw**2 + db**2)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Logging logic
        should_print = (
            i < print_first or i % print_every == 0 or i >= max_iters - print_last
        )

        val_loss = None
        if val_X is not None and val_Y is not None:
            val_loss = compute_loss(val_X, val_Y, w, b)
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

        if verbose and should_print:
            msg = (
                f"Iter {i}: Train Loss = {loss:.5f}, "
                f"w = {w:.5f}, b = {b:.5f}, grad_norm = {grad_norm:.5f}"
            )
            if val_loss is not None:
                msg += f", Val Loss = {val_loss:.5f}"
            print(msg)

        # Early stopping based on patience
        if val_loss is not None and patience_counter >= val_patience:
            if verbose:
                print(
                    f"Early stopping at iter {i} — no val_loss improvement for {val_patience} steps."
                )
            break
    print(f"Final w : {w}, Final b : {b}, loss : {loss}")
    return w, b, loss_history
