Logistic Regression with L1 Penalty
======================================
This class implements L1 (Lasso) regularization on Logistic Regression
using coordinate descent to achieve sparse solutions.
It allows for both in-memory inputs and data that lives in a PySpark RDD.
This implementation is meant to be used for large datasets. For small datasets,
scikit-learn is sufficient.


### Mathematical Formulation
This method is formulated as the minimization of the concave function f.
We can write this as the optimization problem like so:
![Max Eqn] (./images/logit_max.gif)

where the probabilities are defined by the logistic function:

![Prob Eqn] (./images/logit_prob.gif)

The ith index denotes a set of observations with the same x_i, but with
m_i occurrences of this observation, and y_i trials that result in a "success."
The objective is to find the optimal beta cofficients to minimize the
negative log likelihood.

The technique that is utilized is described in section 3 of ["Regularization Paths for Generalized Linear Models via Coordinate Descent"](http://web.stanford.edu/~hastie/Papers/glmnet.pdf).

For an explanation of how this implementation solves the
non-differentiable objective function from the L1 penalty term,
refer to Chapter 13 of _Machine Learning: A Probabilistic
Perspective_ (Murphy, 2012).

Installation
---------------
Download the script `logistic_regression_L1.py`.
You just need to have this file in the same directory that you are running your code.
Alternatively, add the directory location of this file into your Python path in `.bash_profile`:
`export PYTHONPATH=$DIR:$PYTHONPATH` where `$DIR` is the location of this file.


Usage and Example
-----------------
The primary input will be a matrix. If you do not need to use PySpark,
then pass the data in as a numpy array. If pyspark=True, then the
input must be an RDD.

The matrix format for both cases should include observations, a weight
for each row, and the number of successes. It will look something like
this, where each variable listed is an array:
```
[ x1, x2, ... , num_occurrences_of_row, num_successes ]
```

If each row is unweighted, then the second to last column (num_occurrences_of_row)
will be ones. The last column (num_successes) will then be zeros or ones.

The `fit` method will automatically insert an intercept term and return
coefficients that includes this bias.

### The PySpark RDD
For improved performance, the PySpark RDD's partitions should
each hold a single NumPy array.

The `fit` method will check to see if your partition is a list of a
NumPy array. If it is not, it will be converted into this format.

### Example
#### Without PySpark
```python
import numpy as np
from logistic_regression_l1 import LogisticRegressionL1

# x_1, x_2, m, y are predefined arrays
data = np.array([x_1, x_2, m, y]).T
lambda_grid = np.exp(-1*np.linspace(1,17,200))

logit_no_pyspark = LogisticRegressionL1()
logit_no_pyspark.fit(data, lambda_grid, .00000001, False)

# x_3, x_4 are arrays of new observations
new_observations = np.array([x_3, x_4]).T
logit_no_pyspark.predict(new_observations, False)
```

#### With PySpark
```python
import numpy as np
from logistic_regression_l1 import LogisticRegressionL1

# sparkrdd is predefined RDD with same format as non-PySpark case
lambda_grid = np.exp(-1*np.linspace(1,17,200))

logit_pyspark = LogisticRegressionL1()
logit_pyspark.fit(sparkrdd, lambda_grid, .00000001, True)

# sparkrdd_new_obs is the RDD holding the observations to predict
logit_pyspark.predict(sparkrdd_new_obs, True)
```

Regularization Path
-------------------
If you want to use the benefit of the Lasso regularization, then the
output of the `fit` method will return the entire regularization path,
each row corresponding to the lambda parameter that you choose.

The lambda grid will control how you iterate through the L1
penalty/constraint. Note that the step sizes should be small, as this
algorithm utilizes a second-order Taylor approximation.

If you plot the coefficients against the sum of coefficients in each iteration,
the result will be something like this:
![Logistic Path] (./images/logit_path.png =450x)

The most important features will appear in the most constrained iterations.
