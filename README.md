## Logistic Regression with L1 Penalty
This class implements L1 (Lasso) regularization using coordinate
descent to achieve sparse solutions. It allows for both in-memory
inputs and data that lives in a PySpark RDD.

### Mathematical Formulation
This method is formulated as the minimization of the convex function f.
We can write this as the optimization problem like so:
![Min Eqn](/images/logit_min.gif)
$$\underset{\beta}{\text{min}} -log \prod\limits_{
}^i\binom{m_i}{y_i}p_i^{y_i}(1-p_i)^{m_i-y_i} + \lambda||\beta||_1$$

The ith index denotes a set of observations with the same x_i, but with
m_i occurrences of this observation, and y_i trials that result in a "success."
The objective is to find the optimal beta cofficients to miminize the
negative log likelihood.


## Installation
??

## Usage and Example
The primary input will be a matrix. If you do not need to use PySpark,
then pass the data in as a numpy array. If pyspark=True, then the
input must be an RDD.

The matrix format for both cases should include observations, a weight
for each row, and the number of successes. It will look something like
this, where each variable listed is an array:
```
[ x1, x2, ... , num_occurrences_of_row, num_successes ]
```

If each row is unweighted, then the second to last column will be all
zeros.

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

## Regularization Path
If you want to use the benefit of the Lasso regularization, then the
output of the `fit` method will return the entire regularization path,
each row corresponding to the lambda parameter that you choose.

The lambda grid will control how you iterate through the L1
penalty/constraint. Note that the step sizes should be small, as this
algorithm utilizes a second-order Taylor approximation.
