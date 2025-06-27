import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)

#  Problem Statement

# A model for housing price prediction. The training data set contains many examples with 4 features (size, bedrooms, floors and age)
# shown in the table below. Note, in this lab, the Size feature is in sqft while earlier labs utilized 1000 sqft.

# We would like to build a linear regression model using these values so we can then predict the price for
# other houses - say, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.

# ##  Dataset:
# | Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |
# | ----------------| ------------------- |----------------- |--------------|----------------------- |
# | 952             | 2                   | 1                | 65           | 271.5                  |
# | 1244            | 3                   | 2                | 64           | 232                    |
# | 1947            | 3                   | 2                | 17           | 509.8                  |
# | ...             | ...                 | ...              | ...          | ...                    |


file_path = Path(__file__).resolve().parent / "data" / "houses.txt"


def load_house_data():
    data = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=1,
    )  # (99, 5)
    X = data[:, :-1]  # (99, 4)
    Y = data[:, -1:]  # (99, 1)
    # y = data[:, -1]  # (99,) same as data[:, 4]
    return X, Y


# X = (99, 4), Y = (99, 1), W = (4, 1), b = scalar, f_wb = (99, 1), returns a scalar
def compute_cost(X, Y, W, b):
    f_wb = np.dot(X, W) + b
    E = f_wb - Y
    return 0.5 * np.mean(E**2)


def compute_gradient_matrix(X, Y, W, b):
    f_wb = np.dot(X, W) + b
    E = f_wb - Y
    dj_dw = np.dot(X.T, E) / X.shape[0]
    dj_db = np.mean(E)
    return dj_db, dj_dw


def gradient_descent_houses(
    X, Y, W_in, b_in, cost_function, gradient_function, alpha, num_iters
):
    # number of training examples
    m = len(X)
    # To store values at each iteration primarily for graphing later
    hist = {}
    hist["cost"] = []
    hist["params"] = []
    hist["grads"] = []
    hist["iter"] = []

    W = copy.deepcopy(W_in)  # avoid modifying global W within function
    b = b_in

    save_interval = np.ceil(
        num_iters / 10000
    )  # prevent resource exhaustion for long runs
    print(
        f"Iteration Cost          w0        w1        w2        w3        b        djdw0     djdw1     djdw2     djdw3     djdb   "
    )
    print(
        f"----------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|"
    )

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, Y, W, b)

        # Update Parameters using w, b, alpha and gradient
        W = W - alpha * dj_dw
        b = b - alpha * dj_db
        cost = cost_function(X, Y, W, b)

        # Save cost J,w,b at each save interval for graphing
        if i % save_interval == 0:
            hist["cost"].append(cost)
            hist["params"].append([W.squeeze(), b])
            hist["grads"].append([dj_dw.squeeze(), dj_db])
            hist["iter"].append(i)

        # Print first 10 iterations and every 100th iteration
        if i < 10 or i % 100 == 0:
            # print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            print(
                f"{i:9d}  {cost: 0.5e}  {W[0, 0]: 0.1e}  {W[1, 0]: 0.1e}  {W[2, 0]: 0.1e}  {W[3, 0]: 0.1e}  {b: 0.1e}  {dj_dw[0, 0]: 0.1e}  {dj_dw[1,0]: 0.1e}  {dj_dw[2,0]: 0.1e}  {dj_dw[3, 0]: 0.1e}  {dj_db: 0.1e}"
            )

    return W, b, hist


def run_gradient_descent(X, Y, iterations=1000, alpha=1e-6):
    m, n = X.shape
    # initialize parameters
    initial_W = np.zeros((n, 1))
    initial_b = 0
    # run gradient descent
    w_out, b_out, hist_out = gradient_descent_houses(
        X,
        Y,
        initial_W,
        initial_b,
        compute_cost,
        compute_gradient_matrix,
        alpha,
        iterations,
    )
    print(f"w, b found by gradient descent: w: {w_out.squeeze()}, b: {b_out:0.2f} ")
    return (w_out, b_out, hist_out)


# Load the dataset
X_train, Y_train = load_house_data()
X_features = ["size(sqft)", "bedrooms", "floors", "age"]

# Let's view the dataset and it's features by plotting each feature versus price
fix, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], Y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price $(1000's)")
plt.show()
alpha = 9.9e-7

_, _, hist = run_gradient_descent(X_train, Y_train, 10, alpha=9.9e-7)

# Iteration Cost          w0        w1        w2        w3        b        djdw0     djdw1     djdw2     djdw3     djdb
# ----------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
#         0   9.55884e+04   5.5e-01   1.0e-03   5.1e-04   1.2e-02   3.6e-04  -5.5e+05  -1.0e+03  -5.2e+02  -1.2e+04  -3.6e+02
#         1   1.28213e+05  -8.8e-02  -1.7e-04  -1.0e-04  -3.4e-03  -4.8e-05   6.4e+05   1.2e+03   6.2e+02   1.6e+04   4.1e+02
#         2   1.72159e+05   6.5e-01   1.2e-03   5.9e-04   1.3e-02   4.3e-04  -7.4e+05  -1.4e+03  -7.0e+02  -1.7e+04  -4.9e+02
#         3   2.31358e+05  -2.1e-01  -4.0e-04  -2.3e-04  -7.5e-03  -1.2e-04   8.6e+05   1.6e+03   8.3e+02   2.1e+04   5.6e+02
#         4   3.11100e+05   7.9e-01   1.4e-03   7.1e-04   1.5e-02   5.3e-04  -1.0e+06  -1.8e+03  -9.5e+02  -2.3e+04  -6.6e+02
#         5   4.18517e+05  -3.7e-01  -7.1e-04  -4.0e-04  -1.3e-02  -2.1e-04   1.2e+06   2.1e+03   1.1e+03   2.8e+04   7.5e+02
#         6   5.63212e+05   9.7e-01   1.7e-03   8.7e-04   1.8e-02   6.6e-04  -1.3e+06  -2.5e+03  -1.3e+03  -3.1e+04  -8.8e+02
#         7   7.58122e+05  -5.8e-01  -1.1e-03  -6.2e-04  -1.9e-02  -3.4e-04   1.6e+06   2.9e+03   1.5e+03   3.8e+04   1.0e+03
#         8   1.02068e+06   1.2e+00   2.2e-03   1.1e-03   2.3e-02   8.3e-04  -1.8e+06  -3.3e+03  -1.7e+03  -4.2e+04  -1.2e+03
#         9   1.37435e+06  -8.7e-01  -1.7e-03  -9.1e-04  -2.7e-02  -5.2e-04   2.1e+06   3.9e+03   2.0e+03   5.1e+04   1.4e+03
# w, b found by gradient descent: w: [-0.87 -0.   -0.   -0.03], b: -0.00

# It appears the learning rate is too high.  The solution does not converge. Cost is *increasing* rather than decreasing.
# Let's plot the result:


# print(hist["params"])
# ws = np.array([p[0][0] for p in hist["params"]])
# print(ws)
# print(abs(ws[:, 0].min()))
# print(abs(ws[:, 0].max()))


# It appears the learning rate is too high.  The solution does not converge. Cost is *increasing* rather than decreasing.
# Let's plot the result:
def plot_cost_i_w(X, Y, hist):
    ws = np.array([p[0] for p in hist["params"]])
    rng = max(abs(ws[:, 0].max()), abs(ws[:, 0].min()))
    wr = np.linspace(-rng + 0.27, rng + 0.27, 20)
    cst = [compute_cost(X, Y, np.array([w, -32, -67, -1.46])[:, None], 221) for w in wr]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(hist["iter"], hist["cost"])
    ax[0].set_title("Cost vs Iteration")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("Cost")

    ax[1].plot(wr, cst)
    ax[1].set_xlabel("w[0]")
    ax[1].set_ylabel("Cost")
    ax[1].set_title(f"Cost vs w[0]")
    ax[1].plot(ws[:, 0], hist["cost"])
    plt.show()


plot_cost_i_w(X_train, Y_train, hist)

# The plot on the right shows the value of one of the parameters, $w_0$. At each iteration, it is overshooting the
# optimal value and as a result, cost ends up *increasing* rather than approaching the minimum. Note that this is not a
# completely accurate picture as there are 4 parameters being modified each pass rather than just one. This plot is only
# showing $w_0$ with the other parameters fixed at benign values. In this and later plots you may notice the blue and orange
# lines being slightly off.
# Let's try a bit smaller value of learning rate and see what happens.
# alpha = 9e-7
_, _, hist = run_gradient_descent(X_train, Y_train, 10, alpha=9e-7)

# Iteration Cost          w0        w1        w2        w3        b        djdw0     djdw1     djdw2     djdw3     djdb
# ----------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
#         0   6.64616e+04   5.0e-01   9.1e-04   4.7e-04   1.1e-02   3.3e-04  -5.5e+05  -1.0e+03  -5.2e+02  -1.2e+04  -3.6e+02
#         1   6.18990e+04   1.8e-02   2.1e-05   2.0e-06  -7.9e-04   1.9e-05   5.3e+05   9.8e+02   5.2e+02   1.3e+04   3.4e+02
#         2   5.76572e+04   4.8e-01   8.6e-04   4.4e-04   9.5e-03   3.2e-04  -5.1e+05  -9.3e+02  -4.8e+02  -1.1e+04  -3.4e+02
#         3   5.37137e+04   3.4e-02   3.9e-05   2.8e-06  -1.6e-03   3.8e-05   4.9e+05   9.1e+02   4.8e+02   1.2e+04   3.2e+02
#         4   5.00474e+04   4.6e-01   8.2e-04   4.1e-04   8.0e-03   3.2e-04  -4.8e+05  -8.7e+02  -4.5e+02  -1.1e+04  -3.1e+02
#         5   4.66388e+04   5.0e-02   5.6e-05   2.5e-06  -2.4e-03   5.6e-05   4.6e+05   8.5e+02   4.5e+02   1.2e+04   2.9e+02
#         6   4.34700e+04   4.5e-01   7.8e-04   3.8e-04   6.4e-03   3.2e-04  -4.4e+05  -8.1e+02  -4.2e+02  -9.8e+03  -2.9e+02
#         7   4.05239e+04   6.4e-02   7.0e-05   1.2e-06  -3.3e-03   7.3e-05   4.3e+05   7.9e+02   4.2e+02   1.1e+04   2.7e+02
#         8   3.77849e+04   4.4e-01   7.5e-04   3.5e-04   4.9e-03   3.2e-04  -4.1e+05  -7.5e+02  -3.9e+02  -9.1e+03  -2.7e+02
#         9   3.52385e+04   7.7e-02   8.3e-05  -1.1e-06  -4.2e-03   8.9e-05   4.0e+05   7.4e+02   3.9e+02   1.0e+04   2.5e+02
# w, b found by gradient descent: w: [ 7.74e-02  8.27e-05 -1.06e-06 -4.20e-03], b: 0.00


plot_cost_i_w(X_train, Y_train, hist)

# On the left, you see that cost is decreasing as it should. On the right, you can see that $w_0$ is still oscillating around
# the minimum, but it is decreasing each iteration rather than increasing. Note above that `dj_dw[0]` changes sign with each
# iteration as `w[0]` jumps over the optimal value.
# This alpha value will converge. You can vary the number of iterations to see how it behaves.

# Let's try a bit smaller value of learning rate and see what happens.
# alpha = 1e-7

_, _, hist = run_gradient_descent(X_train, Y_train, 10, alpha=1e-7)
# Iteration Cost          w0        w1        w2        w3        b        djdw0     djdw1     djdw2     djdw3     djdb
# ----------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
#         0   4.42313e+04   5.5e-02   1.0e-04   5.2e-05   1.2e-03   3.6e-05  -5.5e+05  -1.0e+03  -5.2e+02  -1.2e+04  -3.6e+02
#         1   2.76461e+04   9.8e-02   1.8e-04   9.2e-05   2.2e-03   6.5e-05  -4.3e+05  -7.9e+02  -4.0e+02  -9.5e+03  -2.8e+02
#         2   1.75102e+04   1.3e-01   2.4e-04   1.2e-04   2.9e-03   8.7e-05  -3.4e+05  -6.1e+02  -3.1e+02  -7.3e+03  -2.2e+02
#         3   1.13157e+04   1.6e-01   2.9e-04   1.5e-04   3.5e-03   1.0e-04  -2.6e+05  -4.8e+02  -2.4e+02  -5.6e+03  -1.8e+02
#         4   7.53002e+03   1.8e-01   3.3e-04   1.7e-04   3.9e-03   1.2e-04  -2.1e+05  -3.7e+02  -1.9e+02  -4.2e+03  -1.4e+02
#         5   5.21639e+03   2.0e-01   3.5e-04   1.8e-04   4.2e-03   1.3e-04  -1.6e+05  -2.9e+02  -1.5e+02  -3.1e+03  -1.1e+02
#         6   3.80242e+03   2.1e-01   3.8e-04   1.9e-04   4.5e-03   1.4e-04  -1.3e+05  -2.2e+02  -1.1e+02  -2.3e+03  -8.6e+01
#         7   2.93826e+03   2.2e-01   3.9e-04   2.0e-04   4.6e-03   1.4e-04  -9.8e+04  -1.7e+02  -8.6e+01  -1.7e+03  -6.8e+01
#         8   2.41013e+03   2.3e-01   4.1e-04   2.1e-04   4.7e-03   1.5e-04  -7.7e+04  -1.3e+02  -6.5e+01  -1.2e+03  -5.4e+01
#         9   2.08734e+03   2.3e-01   4.2e-04   2.1e-04   4.8e-03   1.5e-04  -6.0e+04  -1.0e+02  -4.9e+01  -7.5e+02  -4.3e+01
# w, b found by gradient descent: w: [2.31e-01 4.18e-04 2.12e-04 4.81e-03], b: 0.00

# Cost is decreasing throughout the run showing that $\alpha$ is not too large.
plot_cost_i_w(X_train, Y_train, hist)

# On the left, you see that cost is decreasing as it should. On the right you can see that w[0] is decreasing without
# crossing the minimum. Note above that `dj_w0` is negative throughout the run. This solution will also converge, though
# not quite as quickly as the previous example.


# Feature scaling
def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)  # (n,)
    sigma = np.std(X, axis=0)  # (n,)
    X_norm = (X - mu) / sigma  # (m.n)
    return (X_norm, mu, sigma)


mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
X_mean = X_train - mu
X_norm = X_mean / sigma

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:, 0], X_train[:, 3])
ax[0].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[0].set_title("Unnormalized")
ax[0].axis("equal")

ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
ax[1].set_xlabel(X_features[0])
ax[1].set_ylabel(X_features[3])
ax[1].set_title(r"X - $\mu$")
ax[1].axis("equal")

ax[2].scatter(X_norm[:, 0], X_norm[:, 3])
ax[2].set_xlabel(X_features[0])
ax[2].set_ylabel(X_features[3])
ax[2].set_title(r"Z-score normalized")
ax[2].axis("equal")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu} \nX_sigma = {X_sigma}")
print(f"Peak to peak range by column in raw        X : {np.ptp(X_train, axis=0)}")
print(f"Peak to peak range by column in Normalized X : {np.ptp(X_norm, axis=0)}")

# The peak to peak range of each column is reduced from a factor of thousands to a factor of 2-3 by normalization.

# Norm distribution plot - for later
# Let's re-run our gradient descent algorithm with normalized data.
# Note the **vastly larger value of alpha**. This will speed up gradient descent.

W_norm, b_norm, hist = run_gradient_descent(
    X_norm,
    Y_train,
    1000,
    1.0e-1,
)

# Iteration Cost          w0        w1        w2        w3        b        djdw0     djdw1     djdw2     djdw3     djdb
# ----------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
#         0   5.76170e+04   8.9e+00   3.0e+00   3.3e+00  -6.0e+00   3.6e+01  -8.9e+01  -3.0e+01  -3.3e+01   6.0e+01  -3.6e+02
#         1   4.66388e+04   1.6e+01   5.0e+00   5.5e+00  -1.1e+01   6.9e+01  -7.5e+01  -2.0e+01  -2.2e+01   5.1e+01  -3.3e+02
#         2   3.78419e+04   2.3e+01   6.3e+00   6.8e+00  -1.5e+01   9.8e+01  -6.4e+01  -1.3e+01  -1.4e+01   4.3e+01  -2.9e+02
#         3   3.07712e+04   2.8e+01   7.0e+00   7.5e+00  -1.9e+01   1.2e+02  -5.5e+01  -7.4e+00  -7.1e+00   3.7e+01  -2.6e+02
#         4   2.50738e+04   3.3e+01   7.3e+00   7.7e+00  -2.2e+01   1.5e+02  -4.8e+01  -3.2e+00  -2.1e+00   3.1e+01  -2.4e+02
#         5   2.04735e+04   3.7e+01   7.3e+00   7.6e+00  -2.5e+01   1.7e+02  -4.2e+01   1.6e-02   1.7e+00   2.7e+01  -2.1e+02
#         6   1.67526e+04   4.1e+01   7.1e+00   7.1e+00  -2.7e+01   1.9e+02  -3.7e+01   2.4e+00   4.5e+00   2.3e+01  -1.9e+02
#         7   1.37386e+04   4.4e+01   6.7e+00   6.5e+00  -2.9e+01   2.1e+02  -3.3e+01   4.1e+00   6.6e+00   2.0e+01  -1.7e+02
#         8   1.12939e+04   4.7e+01   6.1e+00   5.6e+00  -3.1e+01   2.2e+02  -3.0e+01   5.4e+00   8.2e+00   1.7e+01  -1.6e+02
#         9   9.30863e+03   5.0e+01   5.5e+00   4.7e+00  -3.2e+01   2.4e+02  -2.7e+01   6.3e+00   9.3e+00   1.5e+01  -1.4e+02
#       100   2.21086e+02   1.1e+02  -2.0e+01  -3.1e+01  -3.8e+01   3.6e+02  -9.2e-01   4.5e-01   5.3e-01  -1.7e-01  -9.6e-03
#       200   2.19209e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -3.0e-02   1.5e-02   1.7e-02  -6.0e-03  -2.6e-07
#       300   2.19207e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -1.0e-03   5.1e-04   5.7e-04  -2.0e-04  -6.9e-12
#       400   2.19207e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -3.4e-05   1.7e-05   1.9e-05  -6.6e-06  -2.7e-13
#       500   2.19207e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -1.1e-06   5.6e-07   6.2e-07  -2.2e-07  -2.6e-13
#       600   2.19207e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -3.7e-08   1.9e-08   2.1e-08  -7.3e-09  -2.6e-13
#       700   2.19207e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -1.2e-09   6.2e-10   6.9e-10  -2.4e-10  -2.6e-13
#       800   2.19207e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -4.1e-11   2.1e-11   2.3e-11  -8.1e-12  -2.7e-13
#       900   2.19207e+02   1.1e+02  -2.1e+01  -3.3e+01  -3.8e+01   3.6e+02  -1.4e-12   7.0e-13   7.6e-13  -2.7e-13  -2.6e-13
# w, b found by gradient descent: w: [110.56 -21.27 -32.71 -37.97], b: 363.16

# The scaled features get very accurate results **much, much faster!**. Notice the gradient of each parameter is tiny by the
# end of this fairly short run. A learning rate of 0.1 is a good start for regression with normalized features.
# Let's plot our predictions versus the target values. Note, the prediction is made using the normalized feature while the plot
# is shown using the original feature values.

# predict target using normalized features
Y_pred = np.dot(X_norm, W_norm) + b_norm
# plot predictions and targets versus original features
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], Y_train, label="target")
    ax[i].scatter(X_train[:, i], Y_pred, color="orange", label="predictions")
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# The results look good. A few points to note:
# - with multiple features, we can no longer have a single plot showing results versus features.
# - when generating the plot, the normalized features were used. Any predictions using the parameters learned from a
# normalized training set must also be normalized.

# **Prediction**
# The point of generating our model is to use it to predict housing prices that are not in the data set. Let's predict the price
# of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old. Recall, that you must normalize the data with the mean and
# standard deviation derived when the training data was normalized.

# First, normalize out example.
X_house = np.array([1200, 3, 1, 40])
X_house_norm = (X_house - X_mu) / X_sigma
print(X_house_norm)
x_house_predict = np.dot(X_house_norm, W_norm) + b_norm
print(
    f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict.squeeze()*1000:0.0f}"
)

# **Cost Contours**
# Another way to view feature scaling is in terms of the cost contours. When feature scales do not match, the plot of cost versus
# parameters in a contour plot is asymmetric.

# In the plot below, the scale of the parameters is matched. The left plot is the cost contour plot of w[0], the square feet
# versus w[1], the number of bedrooms before normalizing the features. The plot is so asymmetric, the curves completing the
# contours are not visible. In contrast, when the features are normalized, the cost contour is much more symmetric. The result
# is that updates to parameters during gradient descent can make equal progress for each parameter.

# contour plots and normal distribution for later
