import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_utils import *
from sklearn.metrics import mean_squared_error

data_path = Path(__file__).resolve().parent / "data"
file_path = Path(data_path) / "ames_housing_subset.csv"
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_housing_data(val_fraction: float = 0.2, random_state: int = 42):
    """Loads the Ames Housing dataset, splits it into training and validation sets,
    and returns features and target names.

    This function reads a CSV file from the predefined `file_path`, assumes the
    last column is the target variable, and the preceding columns are features.
    It then shuffles the data and splits it into training and validation sets
    based on the specified fraction.

    Args:
        val_fraction (float, optional): The fraction of the dataset to be used
                                        as the validation set. Defaults to 0.2 (20%).
        random_state (int, optional): Seed for the random number generator to ensure
                                      reproducible shuffling and splitting of data.
                                      Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features.
            - Y_train (np.ndarray): Training target values.
            - X_val (np.ndarray): Validation features.
            - Y_val (np.ndarray): Validation target values.
            - X_features (list): A list of strings, representing the names of the feature columns.
            - Y_target (str): The name of the target column.

    Raises:
        FileNotFoundError: If the `file_path` does not point to an existing CSV file.
        KeyError: If column names are not as expected (e.g., if the last column
                  isn't the target or if there are no columns).
        Exception: For other unexpected errors during data loading or processing.
    """
    try:
        df = pd.read_csv(file_path)

        # Extract feature and target column names
        # Assumes the last column is the target and all others are features.
        X_features = df.columns[:-1].to_list()
        Y_target = df.columns[-1]

        # Convert DataFrame subsets to NumPy arrays
        X = df[X_features].to_numpy()
        Y = df[[Y_target]].to_numpy()  # Keep Y as 2D array (m, 1) for consistency

        # Shuffle indices for random splitting
        np.random.seed(random_state)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        # Calculate validation set size and split indices
        val_size = int(val_fraction * len(indices))
        val_idx, train_idx = indices[:val_size], indices[val_size:]

        # Split data into training and validation sets
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        return X_train, Y_train, X_val, Y_val, X_features, Y_target

    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        raise  # Re-raise the exception after logging
    except IndexError:  # Catches error if df.columns is empty or has only one column
        logging.error(
            "Error: DataFrame has too few columns to separate features and target."
        )
        raise
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during data loading or splitting: {e}"
        )
        raise  # Re-raise for further handling


# ZScoreScaler class
# - Stores μ (mean) and σ (std dev) for each feature
# - Applies transformation: z = (x - μ) / σ
# - Can also reverse transform if needed (optional)


class ZScoreScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero (in case of constant features)
        self.std_ = np.where(self.std_ == 0, 1, self.std_)

    def transform(self, X):
        assert (
            self.mean_ is not None and self.std_ is not None
        ), "Call fit() before transform()"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        assert (
            self.mean_ is not None and self.std_ is not None
        ), "Call fit() before inverse_transform()"
        return X_scaled * self.std_ + self.mean_


def add_feature(X, features, feature_index, transform_fn, new_feature_name=None):
    """
    Adds a new feature to X by applying a transformation to one column.

    Args:
        X (np.ndarray): Feature matrix, shape (m, n)
        features (List[str]): Feature names (not mutated unless name is added)
        feature_index (int): Index of column to transform
        transform_fn (Callable): A function like lambda x: x ** 2
        new_feature_name (str, optional): Name of new feature. If None, feature list is returned unchanged.

    Returns:
        X_augmented (np.ndarray): Feature matrix with new column
        updated_features (List[str] or None): Updated feature names if name is given, else None
    """
    new_col = transform_fn(X[:, feature_index]).reshape(-1, 1)
    X_augmented = np.hstack([X, new_col])

    if new_feature_name is not None:
        updated_features = features + [new_feature_name]
    else:
        updated_features = None

    return X_augmented, updated_features


class LinearRegressor:
    """
    Linear Regression model using batch gradient descent and vectorized operations.

    Attributes:
        alpha (float): Learning rate for gradient descent.
        num_iters (int): Maximum number of training iterations.
        W (np.ndarray): Weight matrix of shape (n, 1).
        b (float): Bias term.
    """

    def __init__(self, alpha=0.01, num_iters=1000):
        """
        Initializes the linear regressor.

        Args:
            alpha (float): Learning rate.
            num_iters (int): Number of iterations for training.
        """
        self.alpha = alpha
        self.num_iters = num_iters
        self.W = None
        self.b = None

    def initialize_weights(self, n):
        """
        Initializes weights and bias to zeros.

        Args:
            n (int): Number of input features.
        """
        self.W = np.zeros((n, 1))
        self.b = 0

    def compute_cost(self, X, y):
        """
        Computes the mean squared error cost.

        Args:
            X (np.ndarray): Input features of shape (m, n).
            y (np.ndarray): Target values of shape (m, 1).

        Returns:
            float: The MSE cost.
        """
        assert self.W is not None, "Initialize weights first"
        y_pred = np.dot(X, self.W) + self.b
        errors = y_pred - y
        return 0.5 * np.mean(errors**2)

    def gradient_descent_step(self, X, y):
        """
        Performs one step of gradient descent on weights and bias.

        Args:
            X (np.ndarray): Input features of shape (m, n).
            y (np.ndarray): Target values of shape (m, 1).

        Returns:
            Tuple[np.ndarray, float]: Gradients dW and db.
        """
        assert self.W is not None, "Initialize weights first"
        y_pred = np.dot(X, self.W) + self.b
        errors = y_pred - y
        dW = np.dot(X.T, errors) / X.shape[0]
        db = np.mean(errors)
        self.W -= self.alpha * dW
        self.b -= self.alpha * db
        return dW, db

    def fit(self, X, y):
        """
        Trains the model using batch gradient descent.

        Args:
            X (np.ndarray): Input features of shape (m, n).
            y (np.ndarray): Target values of shape (m, 1).

        Returns:
            dict: Dictionary containing training history (costs, parameters, grads, etc.)
        """
        self.initialize_weights(X.shape[1])
        grad_tol = 1e-6
        cost_tol = 1e-6
        prev_cost = None
        hist = {"cost": [], "params": [], "grads": [], "grad_norm": [], "iter": []}
        save_interval = np.ceil(self.num_iters / 10000)

        for i in range(self.num_iters):
            cost = self.compute_cost(X, y)
            dW, db = self.gradient_descent_step(X, y)
            grad_norm = np.sqrt(np.sum(dW**2) + db**2)

            if i % save_interval == 0:
                hist["cost"].append(cost)
                hist["params"].append([copy.deepcopy(self.W.squeeze()), self.b])
                hist["grads"].append([dW.squeeze(), db])
                hist["grad_norm"].append(grad_norm)
                hist["iter"].append(i)

            if i < 10 or i % 100 == 0:
                print(f"{i:9d}  {cost: 0.5e}")

            if prev_cost is not None and abs(cost - prev_cost) < cost_tol:
                print(
                    f"Early stopping at iter {i} — Δcost = {abs(cost - prev_cost):.6e} < {cost_tol}"
                )
                break

            if grad_norm < grad_tol:
                print(
                    f"Early stopping at iter {i} — grad_norm = {grad_norm:.6e} < {grad_tol}"
                )
                break

            prev_cost = cost

        print(
            f"Final w : {copy.deepcopy(self.W.squeeze())}, Final b : {self.b}, cost : {cost}, grad_norm : {grad_norm}"
        )
        return hist

    def predict(self, X):
        """
        Makes predictions using the learned weights.

        Args:
            X (np.ndarray): Scaled input features of shape (m, n).

        Returns:
            np.ndarray: Predicted values of shape (m, 1).
        """
        assert self.W is not None, "Model is not trained yet"
        return np.dot(X, self.W.squeeze()) + self.b


if __name__ == "__main__":

    # Step 1: Load data
    X_train, Y_train, X_val, Y_val, X_features, Y_target = load_housing_data()

    # Step 2: Polynomial Feature Engineering - (Gr Liv Area)^2
    # Add polynomial features to both the training, and the validation set
    gr_liv_area_index = X_features.index("Gr Liv Area")
    transform_fn = lambda x: x**2
    X_train_aug, X_features = add_feature(
        X_train,
        X_features,
        gr_liv_area_index,
        transform_fn,
        new_feature_name="Gr Liv Area^2",
    )
    X_val_aug, _ = add_feature(X_val, X_features, gr_liv_area_index, transform_fn)

    # Step 2 plots - Scatter plots and histograms
    scatter_plots_path = Path(data_path) / "scatter_plots_training_set_augmented.png"
    plot_scatterplots(
        X_train_aug, Y_train, X_features, Y_target, output_path=scatter_plots_path
    )
    histogram_path = Path(data_path) / "histogram_training_set_augmented.png"
    plot_histograms(X_train_aug, X_features, output_path=histogram_path)

    # Step 3: Fit Scalar on training set, Transform both training set and the validation set
    scaler = ZScoreScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_val_scaled = scaler.transform(X_val_aug)

    # Step 3 plots - Scatter plots and histograms on scaled data
    scatter_plots_path = Path(data_path) / "scatter_plots_training_set_scaled.png"
    plot_scatterplots(
        X_train_scaled, Y_train, X_features, Y_target, output_path=scatter_plots_path
    )
    histogram_path = Path(data_path) / "histogram_training_set_scaled.png"
    plot_histograms(X_train_scaled, X_features, output_path=histogram_path)

    # Diagnostics
    print("\n[Diagnostics] Feature Means (should be ~0):")
    print(np.mean(X_train_scaled, axis=0))

    print("\n[Diagnostics] Feature Std Devs (should be ~1):")
    print(np.std(X_train_scaled, axis=0))

    # Step 4: Train the model
    model = LinearRegressor(alpha=0.01, num_iters=1000)
    history = model.fit(X_train_scaled, Y_train)

    # Step 5: Predict on training data
    Y_pred = model.predict(X_val_scaled)

    # Step 6: Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_val, Y_pred, color="slateblue", alpha=0.7, edgecolor="k")
    plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], "r--", lw=2)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Linear Regression Fit on Ames Housing Subset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    rmse = np.sqrt(mean_squared_error(Y_train, Y_pred))
    print(f"\nRMSE on training set: ${rmse:,.2f}")
