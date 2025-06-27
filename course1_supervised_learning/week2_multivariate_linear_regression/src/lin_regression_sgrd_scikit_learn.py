from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=2)

dlc = dict(
    dlblue="#0096ff",
    dlorange="#FF9300",
    dldarkred="#C00000",
    dlmagenta="#FF40FF",
    dlpurple="#7030A0",
)

data_dir = Path(__file__).resolve().parent.parent / "data"
file_path = Path(data_dir) / "houses.txt"


def load_house_data():
    data = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=1,
    )  # (99, 5)
    X = data[:, :-1]  # (99, 4)
    # Y = data[:, :-1]  # (99, 1)
    y = data[:, -1]  # (99,) same as data[:, 4]
    return X, y


# Gradient Descent
# Scikit-learn has a gradient descent regression model
# [sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#examples-using-sklearn-linear-model-sgdregressor).
# Like our previous implementation of gradient descent, this model performs best with normalized inputs.
# [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
# will perform z-score normalization as in a previous lab. Here it is referred to as 'standard score'.

# Load the data set
X_train, y_train = load_house_data()
X_features = ["size(sqft)", "bedrooms", "floors", "age"]
print(y_train)


# Scale/normalize the training data
scalar = StandardScaler()
X_norm = scalar.fit_transform(X_train)
# print(scalar.mean_)
# print(scalar.scale_)
# print(np.mean(X_train, axis=0))
# print(np.std(X_train, axis=0))
print(f"Peak to peak range by column in raw            X: {np.ptp(X_train, axis=0)}")
print(f"Peak to peak range by column in normalized     X: {np.ptp(X_norm, axis=0)}")

# Create and fit the regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(
    f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}"
)

# ### View parameters
# Note, the parameters are associated with the *normalized* input data. The fit parameters are very close to those found in
# the previous lab with this data.

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"Model parameters                    w: {w_norm}, b: {b_norm}")
print("model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# ### Make predictions
# Predict the targets of the training data. Use both the `predict` routine and compute using $w$ and $b$.

# make a prediction using sgdr.predict()
y_pred_sgdr = sgdr.predict(X_norm)
# make a prediction using w,b.
y_pred = np.dot(X_norm, w_norm) + b_norm
print(
    f"prediction using np.dot() and sgdr.predict match: {(y_pred_sgdr == y_pred).all()}"
)

print(f"Prediction on training set:\n{y_pred[:4]}")
print(f"Target values:\n{y_train[:4]}")

### Plot Results
# Let's plot the predictions versus the target values.

# plot predictions and targets vs original features
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label="target")
    ax[i].set_xlabel("X_features[i]")
    ax[i].scatter(X_train[:, i], y_pred, color=dlc["dlorange"], label="predict")
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
