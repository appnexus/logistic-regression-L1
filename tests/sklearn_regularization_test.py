import numpy as np
from logistic_regression_L1 import LogisticRegressionL1

from sklearn import linear_model
from sklearn.svm import l1_min_c
from sklearn.datasets import make_classification

def regularization_test(weight_list=[0.5, 0.5]):

    num_samples = 10
    num_feat = 5

    X, y = make_classification(n_samples=num_samples, n_features=num_feat, n_informative=3,
                                   n_classes=2, random_state=0, weights=weight_list)
    matrix = np.zeros((num_samples, num_feat + 2))
    matrix[:,:-2] = X
    matrix[:, -2] = np.ones(num_samples)
    matrix[:, -1] = y

    # Betas to test
    logitfitL1 = LogisticRegressionL1()
    lambda_grid = np.exp(-1*np.linspace(1,17,200))
    path = logitfitL1.fit(matrix, lambda_grid)

    # Sklearn
    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)

    # Computing regularization path using sklearn
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())

    skbetas = np.append(clf.intercept_[0], clf.coef_)
    np.testing.assert_almost_equal(skbetas, logitfitL1.coef_, 1)
    
regularization_test()
regularization_test([0.6, 0.4])