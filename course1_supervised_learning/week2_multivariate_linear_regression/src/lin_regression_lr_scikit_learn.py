# ## Goals
# In this lab you will:
# - Utilize  scikit-learn to implement linear regression using a close form solution based on the normal equation
# ## Tools
# You will utilize functions from scikit-learn as well as matplotlib and NumPy.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=2)


# # Linear Regression, closed-form solution
# Scikit-learn has the [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) which implements a closed-form linear regression.

# Let's use the data from the early labs - a house with 1000 square feet sold for \\$300,000 and a house with 2000 square feet sold for \\$500,000.

# | Size (1000 sqft)     | Price (1000s of dollars) |
# | ----------------| ------------------------ |
# | 1               | 300                      |
# | 2               | 500                      |

### Load the data set

X_train = np.array([1.0, 2.0])
Y_train = np.array([300, 500])

### Create and fit the model
# The code below performs regression using scikit-learn.
# The first step creates a regression object.
# The second step utilizes one of the methods associated with the object, `fit`. This performs regression, fitting the parameters
#  to the input data. The toolkit expects a two-dimensional X matrix.

linear_model = LinearRegression()
linear_model.fit(X_train.reshape(-1, 1), Y_train)

### View Parameters
# The $\mathbf{w}$ and $\mathbf{b}$ parameters are referred to as 'coefficients' and 'intercept' in scikit-learn.

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")

### Make Predictions

# Calling the `predict` function generates predictions.
y_pred = linear_model.predict(X_train.reshape(-1, 1))
print("Prediction on training set:", y_pred)
X_test = np.array([[1200]])
print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test)[0]:.2f}")

## Second Example
# The second example is from an earlier lab with multiple features. The final parameter values and predictions are very close
# to the results from the un-normalized 'long-run' from that lab. That un-normalized run took hours to produce results, while
# this is nearly instantaneous. The closed-form solution work well on smaller data sets such as these but can be computationally
# demanding on larger data sets.
# >The closed-form solution does not require normalization.
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


# load the dataset
X_train, y_train = load_house_data()
X_features = ["size(sqft)", "bedrooms", "floors", "age"]

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

b = linear_model.intercept_
w = linear_model.coef_
print(f"w={w:}, b={b:0.2f}")

print(f"Predictions on training set: \n {linear_model.predict(X_train)[:4]}")
print(f"prediction using w,b:\n {(np.dot(X_train, w) + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

X_house = np.array([1200, 3, 1, 40]).reshape(-1, 4)
X_house_predict = linear_model.predict(X_house)[0]
print(
    f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${X_house_predict*1000:0.2f}"
)

## Congratulations!
# In this lab you:
# - utilized an open-source machine learning toolkit, scikit-learn
# - implemented linear regression using a close-form solution from that toolkit
