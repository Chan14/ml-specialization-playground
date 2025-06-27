import copy
import math

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)


def zscore_normalize_features(X):
    X_mu = np.mean(X, axis=0)
    X_sigma = np.std(X, axis=0)
    X_norm = (X - X_mu) / X_sigma
    return X_norm, X_mu, X_sigma


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


def run_gradient_descent_feng(X, Y, iterations=1000, alpha=1e-6):
    m, n = X.shape
    initial_W = np.zeros((n, 1))
    initial_b = 0
    W_out, b_out, hist_out = gradient_descent(
        X,
        Y,
        initial_W,
        initial_b,
        compute_cost,
        compute_gradient_matrix,
        alpha,
        iterations,
    )
    print(f"w,b found by gradient descent: w: {W_out.squeeze()}, b: {b_out:0.4f}")
    return W_out, b_out


def gradient_descent(
    X, Y, W_in, b_in, cost_function, gradient_function, alpha, num_iters
):
    m = len(X)
    # An array to store values at each iteration primarily for graphing later
    hist = {}
    hist["cost"] = []
    hist["params"] = []
    hist["grads"] = []
    hist["iter"] = []
    W = copy.deepcopy(W_in)
    b = b_in
    save_interval = np.ceil(
        num_iters / 10000
    )  # prevent resource exhaustion for long runs

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, Y, W, b)

        W = W - alpha * dj_dw
        b = b - alpha * dj_db
        cost = cost_function(X, Y, W, b)

        # Save cost J,w,b at each save interval for graphing
        if i % save_interval == 0:
            hist["cost"].append(cost)
            hist["params"].append([W.squeeze(), b])
            hist["grads"].append([dj_dw.squeeze(), dj_db])
            hist["iter"].append(i)

        # Print first 10 iterations and every 10th iteration
        if i % 100 == 0:
            print(f"Iteration {i:9d}, Cost: {cost:0.5e}")
    return W, b, hist  # return w,b and history for graphing


x = np.arange(0, 20, 1)  # (20,)
y = 1 + x**2  # (20, )
X = x[:, None]  # (20, 1)
Y = y[:, None]

model_w, model_b = run_gradient_descent_feng(X, Y, iterations=1000, alpha=1e-2)
Y_pred = X @ model_w + model_b

plt.scatter(X, Y, marker="x", color="r", label="Actual Value")
plt.title("No Feature Engineering")
plt.plot(X, Y_pred, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Well, as expected, not a great fit. What is needed is something like $y= w_0x_0^2 + b$, or a **polynomial feature**.
# To accomplish this, you can modify the *input data* to *engineer* the needed features. If you swap the original data with a
# version that squares the $x$ value, then you can achieve $y= w_0x_0^2 + b$. Let's try it. Swap `X` for `X**2` below:

# create target data
x = np.arange(0, 20, 1)  # (20,)
y = 1 + x**2  # (20, )

# Engineer feature
X = x**2
X = X[:, None]
Y = y[:, None]

model_w, model_b = run_gradient_descent_feng(X, Y, iterations=10000, alpha=1e-5)
Y_pred = X @ model_w + model_b
plt.scatter(x, Y, marker="x", color="r", label="Actual Value")
plt.title("Added x**2 feature")
plt.plot(x, Y_pred, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Great! near perfect fit. Notice the values of $\mathbf{w}$ and b printed right above the graph: `w,b found by gradient
# descent: w: [1.], b: 0.0490`. Gradient descent modified our initial values of $\mathbf{w},b $ to be (1.0,0.049) or a model
# of $y=1*x_0^2+0.049$, very close to our target of $y=1*x_0^2+1$. If you ran it longer, it could be a better match.

# ### Selecting Features
# Above, we knew that an $x^2$ term was required. It may not always be obvious which features are required. One could add a
# variety of potential features to try and find the most useful. For example, what if we had instead tried :
# $y=w_0x_0 + w_1x_1^2 + w_2x_2^3+b$ ?


x = np.arange(0, 20, 1)
y = x**2
Y = y[:, None]
X = np.c_[x, x**2, x**3]


model_w, model_b = run_gradient_descent_feng(X, Y, iterations=10000, alpha=1e-7)
Y_pred = X @ model_w + model_b
plt.scatter(x, Y, marker="x", color="r", label="Actual Value")
plt.title("x, x**2, x**3 features")
plt.plot(x, Y_pred, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Note the value of $\mathbf{w}$, `[0.08 0.54 0.03]` and b is `0.0106`.This implies the model after fitting/training is:
# $$ 0.08x + 0.54x^2 + 0.03x^3 + 0.0106 $$
# Gradient descent has emphasized the data that is the best fit to the $x^2$ data by increasing the $w_1$ term relative to the
# others.  If you were to run for a very long time, it would continue to reduce the impact of the other terms.
# Gradient descent is picking the 'correct' features for us by emphasizing its associated parameter

# Let's review this idea:
# - Intially, the features were re-scaled so they are comparable to each other
# - less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or very close to
# zero, the associated feature useful in fitting the model to the data.
# - above, after fitting, the weight associated with the $x^2$ feature is much larger than the weights for $x$ or $x^3$ as it
# is the most useful in fitting the data.

### An Alternate View
# Above, polynomial features were chosen based on how well they matched the target data. Another way to think about this is
# to note that we are still using linear regression once we have created new features. Given that, the best features will be
# linear relative to the target. This is best understood with an example.

x = np.arange(0, 20, 1)
y = x**2
Y = y[:, None]
X = np.c_[x, x**2, x**3]
X_features = ["x", "x^2", "x^3"]

fix, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:, i], Y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Y")
plt.show()

# Above, it is clear that the $x^2$ feature mapped against the target value $y$ is linear. Linear regression can then easily
# generate a model using that feature.
# ### Scaling features
# As described in the last lab, if the data set has features with significantly different scales, one should apply feature
# scaling to speed gradient descent. In the example above, there is $x$, $x^2$ and $x^3$ which will naturally have very
# different scales. Let's apply Z-score normalization to our example.

x = np.arange(0, 20, 1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in raw              X : f{np.ptp(X, axis=0)}")
X_norm, X_mu, X_sigma = zscore_normalize_features(X)
print(f"Peak to Peak range by column in normalized       X : f{np.ptp(X_norm, axis=0)}")

# Now we can try again with a more aggressive value of alpha:

x = np.arange(0, 20, 1)
y = x**2
Y = y[:, None]
X = np.c_[x, x**2, x**3]
X_norm, X_mu, X_sigma = zscore_normalize_features(X)
X_features = ["x", "x^2", "x^3"]

model_w, model_b = run_gradient_descent_feng(X_norm, Y, iterations=100000, alpha=1e-1)
Y_pred = X_norm @ model_w + model_b
plt.scatter(x, Y, marker="x", color="r", label="Actual Value")
plt.title("Normalized x x**2, x**3 feature")
plt.plot(x, Y_pred, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Feature scaling allows this to converge much faster.
# Note again the values of $\mathbf{w}$. The $w_1$ term, which is the $x^2$ term is the most emphasized.
# Gradient descent has all but eliminated the $x^3$ term.

# ### Complex Functions
# With feature engineering, even quite complex functions can be modeled:

x = np.arange(0, 20, 1)
y = np.cos(x / 2)
Y = y[:, None]
X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X_norm, X_mu, X_sigma = zscore_normalize_features(X)


model_w, model_b = run_gradient_descent_feng(X_norm, Y, iterations=1000000, alpha=1e-1)
Y_pred = X_norm @ model_w + model_b
plt.scatter(x, Y, marker="x", color="r", label="Actual Value")
plt.title("Normalized sum(x^p) for p = 0, 1,2 .. 13")
plt.plot(x, Y_pred, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#
## Congratulations!
# In this lab you:
# - learned how linear regression can model complex, even highly non-linear functions using feature engineering
# - recognized that it is important to apply feature scaling when doing feature engineering
