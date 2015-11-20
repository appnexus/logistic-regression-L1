from setuptools import setup

setup(
    name='logistic_regression_L1',
    version='0.1',
    author='Stephanie Tzeng',
    author_email='stzeng@appnexus.com',
    packages=['logistic_regression_L1'],
    url='https://github.com/appnexus/logistic_regression_L1',
    description='Logistic Regression with L1 Penalty',
    long_description='',
    install_requires=[
        'numpy'
    ],
    test_requires=[
        'nose',
        'coverage',
        'unittest2'
    ],
)
