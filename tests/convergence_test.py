"""
Convergence tests, without spark
"""

import numpy as np
from logistic_regression_L1 import LogisticRegressionL1

def prob(X, betas):
    """
    Calculate probability given beta coefficient (sigmoid function)

    Params
    ------
    X : matrix of observations, not including bias
    betas : array of coefficients, includes bias

    Returns
    -------
    prob : array, probability for each observation
    """
    betas = betas * 1.
    X = np.insert(X, 0, 1., axis=1) * 1.
    power = X.dot(betas)

    return np.exp(power) / (1 + np.exp(power))

def create_random_observations(num_obs, num_feat, betas):
    """
    Create random observations, X, weights, and y (successes)

    Params
    ------
    num_obs : int
    num_feat : int (doesn't include bias)
    betas : list of beta coefficients, includes bias

    Returns
    -------
    matrix : 2d np array formatted as [X columns, m, y]
    """
    matrix = np.zeros((num_obs, num_feat + 2))

    for i in xrange(num_feat):
        matrix[:,i] = np.random.randint(0, 100, size=num_obs) * 1.0 / 100

    m = np.random.randint(500, 1000, size = num_obs)

    X = matrix[:, :-2]
    P = prob(X, np.array(betas))
    y = P * m

    matrix[:, -2] = m
    matrix[:, -1] = y

    return matrix

def explode_matrix(data):

    rows, col = data.shape

    m = data[:, -2]
    y = data[:, -1]

    new_matrix = np.zeros((int(m.sum()), col))

    # make m rows, y of which are successes
    rownum = 0
    for i in xrange(len(data)):
        x = data[i, :-2]

        for j in xrange(int(data[i, -2])):
            # iterate through all trials per row
            new_matrix[rownum, :-2] = x

            if j < int(data[i, -1]):
                new_matrix[rownum, -1] = 1
            rownum += 1

    new_matrix[:, -2] = np.ones(int(m.sum()))

    return new_matrix

# Run test for 2 features, 100 observations
betas = [5., 0.3, 1.]

logitfitL1 = LogisticRegressionL1()
matrix = create_random_observations(100, 2, betas)
lambda_grid = np.exp(-1*np.linspace(1,17,200))
path = logitfitL1.fit(matrix, lambda_grid)

np.testing.assert_almost_equal(np.array(betas), logitfitL1.coef_, 2)
obs = matrix[:, :-2]

# Run test for prediction function
predictions = np.divide(matrix[:, -1], matrix[:, -2])
np.testing.assert_almost_equal(logitfitL1.predict(obs), predictions, 2)

# Check results with scikit learn
from sklearn.linear_model.logistic import LogisticRegression

betas = [0.001, 0.07, 0.4]
matrix = create_random_observations(200, 2, betas)
new_matrix = explode_matrix(matrix)
X = new_matrix[:,:-2]
y = new_matrix[:, -1]


lib = LogisticRegression(fit_intercept=True)
lib.fit(X, y)

path = logitfitL1.fit(new_matrix, lambda_grid)

skbetas = np.append(lib.intercept_[0], lib.coef_)
np.testing.assert_almost_equal(skbetas, logitfitL1.coef_, 2)
