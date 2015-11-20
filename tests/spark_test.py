"""
Convergence tests, with spark
"""

import unittest2
import numpy as np
from logistic_regression_L1 import LogisticRegressionL1
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


class LogisticRegressionL1TestCase(unittest2.TestCase):
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
        self.logitfitL1.fit(matrix_RDD, self.lambda_grid, pyspark=True)
        np.testing.assert_almost_equal(np.array(betas), self.logitfitL1.coef_, 2)

    def test_prediction(self):
        # Test predict function
        betas = [5., 0.3, 1.]
        matrix = create_random_observations(100, 2, betas)
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
