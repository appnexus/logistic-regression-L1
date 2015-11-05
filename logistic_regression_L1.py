"""
Logistic Regression with L1 Penalty (Lasso)
"""

import numpy as np
import math
import copy
import operator as op

def list_add(x, y):
    """
    For adding elements of a list (of arrays) in PySpark.
    """
    for i in xrange(len(x)):
        x[i] += y[i]
    return x

def to_np_array(lol):
    """
    Converts list of lists from PySpark partition
    into a np.array

    Params
    ------
    lol : list of lists

    Returns
    -------
    lmfao : list of np.array
    """
    lines = list(lol)
    D = len(lines[0])
    mfao = np.zeros((len(lines), D))

    for i in xrange(len(lines)):
        mfao[i] = np.array(lines[i])
    return [mfao]

class LogisticRegressionL1():

    def __init__(self):
        self.coef_ = np.array([])

    def __calc_prob(self, X, betas):
        """
        Calculate probability.
        Note that we are assuming that log odds fit a linear model.

        Params
        ------
        matrix : matrix with bias term, X, weights, and responses (y)
        betas : array of coefficients, includes bias

        Returns
        -------
        prob : array, probability for each observation
        """
        betas = betas * 1.
        X = X * 1.
        power = X.dot(betas)

        return np.exp(power) / (1 + np.exp(power))

    def __calc_new_weights(self, matrix, betas):
        """
        New form of weights for "linear regression"
         when rewriting the logistic formula.

        Params
        ------
        matrix : np array with bias, X, weights (num trials), and responses
        betas : array of coefficients, includes bias

        Returns
        -------
        w : array, new weight per observation
        """
        X = matrix[:, :-2]
        weight = matrix[:, -2]
        prob = self.__calc_prob(X, betas)

        return weight * prob * (1 - prob)

    def __calc_z_response(self, matrix, old_betas):
        """
        Is the "response" for all i observations that reformulates the problem
        into a linear regression.

        Is calculated by using the old beta and old p.

        Params
        ------
        matrix : np array with bias, X, weights (num trials), and responses
        old_betas : array, betas from previous lambda iteration

        Returns
        -------
        z : array, new "response" for the observations
        """
        X = matrix[:, :-2]
        weight = matrix[:, -2]
        y = matrix[:, -1]

        prob = self.__calc_prob(X, old_betas)
        new_weights = self.__calc_new_weights(matrix, old_betas)

        return X.dot(old_betas) + \
                (y - weight * prob) / new_weights

    def __calc_aj(self, matrix, new_weights, j):
        """
        a term of jth feature, where log likelihood is written as
         -aj*bj + cj.
        Written in this format to solve l1 like linear regression.

        Params
        ------
        matrix : np array with bias, X, weights (num trials), and responses
        new_weights : result of __calc_new_weights()

        Returns
        -------
        aj : float
        """
        xj = matrix[:, j]

        return (xj ** 2).dot(new_weights)


    def __get_c_precalculated_terms(self, matrix, old_betas):
        """
        Generates pre-calculated c terms that are used for each lambda
        iteration.

        Params
        ------
        matrix : np array with bias, X, weights (num trials), and responses
        old_betas : old array of coefficients, includes bias

        Returns
        -------
        c1: array of feature length. Is first term of c,
            wi*wij*yi summed over all i
        c2_matrix: matrix of dimensions feature length x feature length
            Is part of second term of c (betas included later).
            Formed as wi*xij*xij, summed over all i, where the columns
            are the features that are multiplied by beta_noj
            and the coordinate descent algo iterates over the rows.
        """
        X = matrix[:, :-2]
        new_weights = self.__calc_new_weights(matrix, old_betas)

        z = self.__calc_z_response(matrix, old_betas)

        c1 = np.sum(np.multiply(X.T, np.multiply(new_weights, z)).T, axis=0)
        c2_first_mult = np.multiply(X.T, new_weights)
        c2_matrix = np.dot(c2_first_mult, X)

        return c1, c2_matrix

    def __calc_cj(self, c1, c2_matrix, betas, j):
        """
        Calculates the c term of the jth feature, using pre-calculated
        c values for a particular lambda iteration.

        Params
        ------
        c1: array, result of __get_c_precalculated_terms()
        c2_matrix : matrix of dimensions feature length x feature length
            Result of __get_c_precalculated_terms()
            where the columns are features, and the rows are the sum
            of observations for that feature.

        Returns
        -------
        cj : float
        """
        betas_noj = np.delete(betas, j, 0)

        return c1[j] - np.dot(betas_noj, np.delete(c2_matrix[j], j, 0))

    def __calc_a_c_array(self, matrix, old_betas, total_trials):
        """
        Calculate a array and initialized c array to store values
        for each feature

        Params
        ------
        matrix : np array with bias, X, weights (num trials), and responses
        old_betas : array of last iteration of betas
        total_trials : float

        Returns
        -------
        list of:
            a_array : array of a, each element corresponding to a feature
            c1_array : array
            c2_matrix : matrix
        """
        X = matrix[:, :-2]
        a = []

        c1, c2_matrix = self.__get_c_precalculated_terms(matrix,
                                                         old_betas)
        new_weight = self.__calc_new_weights(matrix, old_betas)

        for j in xrange(len(old_betas)):
            a.append(self.__calc_aj(matrix, new_weight, j))

        a_array = np.array(a) / total_trials
        c1_array = c1 / total_trials
        c2_matrix = c2_matrix / total_trials

        return [a_array, c1_array, c2_matrix]

    def __calc_betaj(self, aj, cj, lam, j):
        """
        We are solving each feature at a time.
        The bias term will have no regularization.

        Params
        ------
        aj : float
        cj : float
        lam : lambda parameter
        j : feature

        Returns
        -------
        bj : float
        """
        if j == 0:
            lam = 0
        if cj < -lam:
            return (cj + lam) / aj
        elif cj > lam:
            return (cj - lam) / aj
        else:
            return 0

    def __calculate_optimal_betas(self, matrix, old_betas, lam,
            total_trials, precision, pyspark):
        """
        For each lambda iteration, determine the converged betas.
        It exits the loop once the pct change of betas is less than the precision.

        Params
        -------
        matrix : np array with bias, X, weights (num trials), and responses
        old_betas : array
        lam : float
        total_trials : float
        precision : float, the amount of precision to use in iterating over betas
            in coordinate descent algorithm
        pyspark : boolean

        Returns
        -------
        new_betas : array
        """

        if pyspark == False:
            a_c_values = self.__calc_a_c_array(matrix,
                                            old_betas, total_trials)
        else:
            a_c_values = matrix.map(lambda x:
                self.__calc_a_c_array(x, old_betas, total_trials)).reduce(list_add)

        a_array = a_c_values[0]
        c1 = a_c_values[1]
        c2_matrix = a_c_values[2]

        # betas change one j at a time
        new_betas = copy.deepcopy(old_betas)
        beta_pct_diff = float("inf")

        while beta_pct_diff > precision:

            for j in xrange(len(old_betas)):
                cj = self.__calc_cj(c1, c2_matrix, new_betas, j)
                new_betas[j] = self.__calc_betaj(a_array[j], cj, lam, j)

            beta_pct_diff = (max(np.abs(new_betas - old_betas)) * 1. /
                                np.sum(np.abs(new_betas)))
            old_betas = copy.deepcopy(new_betas)

        return new_betas

    def fit(self, matrix, lambda_grid=np.exp(-1*np.linspace(1,17,300)),
            precision = .00000001, pyspark=False):
        """
        Calculate the full path of betas, given the data and lambdas
        over which to iterate.
        The beta_path is returned, but the coefficient uses the last element.
        Note that the beta_path is not standardized.

        Params
        ------
        matrix : numpy.array
            or if pyspark=True, RDD whose partitions are lists of a numpy array.
            If a list of lists are inputted for the partition, it will be
            converted into the list of a numpy array.

            First set of columns are feature data, with NO bias term.
            Next set of columns is weights, or number of trials.
            Last set of columns is responses
             like this: np.array([X, weights, y])
        lambda_grid : array of all lambdas to iterate.
            Starts at lambda where non-bias betas are all zero and have not
            "popped out yet," but decrease to a small mumber where essentially
            there is no regularization.
        precision : float, for convergence in coordinate descent
        pyspark : boolean

        Returns
        -------
        beta_path : np array, of arrays of betas for each lambda iteration.
                Starts with largest penalty, or when lambda is largest.
                Ends with unpenalized case (lambda is tiny)
        """
        if pyspark == False:
            # Add in bias
            matrix = np.insert(matrix, 0, 1., axis=1) * 1.
            total_trials = np.sum(matrix[:, -2]) * 1.
            total_successes = np.sum(matrix[:, -1]) * 1.
            num_feat = matrix.shape[1] - 2 # num_feat includes bias

        else:
            if not type(matrix.first()) == numpy.ndarray:
                matrix = matrix.mapPartitions(to_np_array).cache()

            matrix = matrix.map(lambda x: np.insert(x, 0, 1., axis=1) * 1.)
            total_trials = matrix.map(lambda x:
                                      np.sum(x[:, -2]) * 1.).reduce(op.add)
            total_successes = matrix.map(lambda x:
                                         np.sum(x[:, -1]) * 1.).reduce(op.add)
            num_feat = matrix.first().shape[1] - 2

        global_rate = total_successes / total_trials
        beta_guess = np.zeros(num_feat)
        beta_guess[0] = math.log(global_rate / (1.0 - global_rate))

        beta_path = np.zeros((len(lambda_grid), len(beta_guess)))

        for i in xrange(len(lambda_grid)):
            beta_guess = self.__calculate_optimal_betas(matrix, beta_guess,
                            lambda_grid[i], total_trials, precision,
                            pyspark)

            beta_path[i, :] = beta_guess

        self.coef_ = beta_path[-1, :]

        return beta_path

    def predict(self, X, pyspark=False):
        """
        Logistic Regression assumes that the log of the odds fits a linear
        model. We can rewrite the probability in terms of the linear model.

        P = exp(beta.T dot x_k) / (1 + exp(beta.T dot x_k)

        Params
        ------
        X : ndarray, data of shape [n_samples, n_features]
            if pyspark=True, X is an RDD of the observational data.
        pyspark : boolean

        Returns
        -------
        C: array (if pyspark=False), shape [n_samples] of probabilities
            between 0, 1
           if pyspark=True, returns RDD of original dataset with a new column of
            predictions, like [ original_data, prediction ]
        """
        if pyspark == False:
            # Add in bias
            X = np.insert(X, 0, 1., axis=1) * 1.
            return self.__calc_prob(X, self.coef_)

        else:
            # Add in bias
            X = X.map(lambda x: np.insert(x, 0, 1., axis=1) * 1.)
            return X.map(lambda x: X + [self.__calc_prob(x, self.coef_)])
