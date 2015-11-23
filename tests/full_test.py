"""
Unit tests
"""

import unittest2
import numpy as np
from logistic_regression_L1 import LogisticRegressionL1
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import l1_min_c
from sklearn.datasets import make_classification
from pyspark import SparkContext
from pyspark import SparkConf

sc = SparkContext("local[4]")

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

class LogisticRegressionL1NumPyTestCase(unittest2.TestCase):
    def setUp(self):
        super(LogisticRegressionL1NumPyTestCase, self).setUp()
        self.logitfitL1 = LogisticRegressionL1()
        self.lambda_grid = np.exp(-1*np.linspace(1, 17, 200))

    def test_convergence(self):
        # Check convergence and fit

        # Run test for 2 features, 100 observations
        betas = [5., 0.3, 1.]
        matrix = create_random_observations(100, 2, betas)

        path = self.logitfitL1.fit(matrix_RDD, self.lambda_grid)
        np.testing.assert_almost_equal(np.array(betas), self.logitfitL1.coef_, 2)

    def test_prediction(self):
        # Test predict function
        betas = [5., 0.3, 1.]
        matrix = create_random_observations(100, 2, betas)
        obs = matrix[:, :-2]

        path = self.logitfitL1.fit(matrix_RDD, self.lambda_grid)
        predictions = np.divide(matrix[:, -1], matrix[:, -2])
        np.testing.assert_almost_equal(self.logitfitL1.predict(obs), predictions, 2)

    def test_scikit_learn_exploded_data(self):
        # Check results with scikit learn

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

    def test_regularization_path(self):
        # Check results using logistic path
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
        clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        coefs_ = []
        for c in cs:
            clf.set_params(C=c)
            clf.fit(X, y)
            coefs_.append(clf.coef_.ravel().copy())

        skbetas = np.append(clf.intercept_[0], clf.coef_)
        np.testing.assert_almost_equal(skbetas, logitfitL1.coef_, 1)

class LogisticRegressionL1SparkTestCase(unittest2.TestCase):
    @classmethod
    def setUpClass(cls):
        class_name = cls.__name__
        cls.sc = SparkContext(cls.getMaster(), appName=class_name)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()
        # To avoid Akka rebinding to the same port, since it doesn't unbind
        # immediately on shutdown
        cls.sc._jvm.System.clearProperty("spark.driver.port")

    def setUp(self):
        super(LogisticRegressionL1TestCase, self).setUp()
        self.logitfitL1 = LogisticRegressionL1()
        self.lambda_grid = np.exp(-1*np.linspace(1, 17, 200))

    def test_convergence(self):
        # Check convergence and fit

        # Run test for 2 features, 100 observations
        betas = [5., 0.3, 1.]
        matrix = create_random_observations(100, 2, betas)

        matrix_RDD = sc.parallelize(matrix)
        path = self.logitfitL1.fit(matrix_RDD, self.lambda_grid, pyspark=True)
        np.testing.assert_almost_equal(np.array(betas), self.logitfitL1.coef_, 2)

    def test_prediction(self):
        # Test predict function
        betas = [5., 0.3, 1.]
        matrix = create_random_observations(100, 2, betas)
        path = self.logitfitL1.fit(matrix_RDD, self.lambda_grid, pyspark=True)

        obs = matrix[:, :-2]
        predictions = np.divide(matrix[:, -1], matrix[:, -2])
        np.testing.assert_almost_equal(self.logitfitL1.predict(obs), predictions, 2)

    def test_user_inputs__list(self):
        # Test lists of lists as input
        betas = [5., 0.3, 1.]
        matrix = create_random_observations(100, 2, betas)
        matrix_RDD_lol = sc.parallelize([list(i) for i in matrix])
        self.logitfitL1.fit(matrix_RDD_lol, self.lambda_grid, pyspark=True)
        np.testing.assert_almost_equal(np.array(betas), self.logitfitL1.coef_, 2)

    def test_user_inputs__numpy(self):
        # Test list of np.arrays as input
        betas = [5., 0.3, 1.]
        matrix = create_random_observations(100, 2, betas)
        matrix_RDD_mol = sc.parallelize(list(matrix))
        self.logitfitL1.fit(matrix_RDD_mol, self.lambda_grid, pyspark=True)
        np.testing.assert_almost_equal(np.array(betas), self.logitfitL1.coef_, 2)

    def test_non_positive_betas(self):
        # Test negative and zero betas
        betas = [.3, -8, 0, 1.]
        matrix = create_random_observations(100, 3, betas)
        matrix_RDD = sc.parallelize(matrix)
        self.logitfitL1.fit(matrix_RDD, self.lambda_grid, pyspark=True)
        np.testing.assert_almost_equal(np.array(betas), self.logitfitL1.coef_, 3)


if __name__ == "__main__":
    unittest2.main()
