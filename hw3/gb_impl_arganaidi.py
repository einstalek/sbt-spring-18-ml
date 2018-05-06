#coding=utf-8

from scipy.optimize import minimize, fmin_slsqp
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 4}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05


def loss_function(y, h):
    return sum(np.log(1 + np.exp(-2*y*h))) / len(y)


def antigrad(y, h):
    return 2*y / (1 + np.exp(2*y*h))

class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        self.estimators = []
        self.weights = []
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict, random_state=1)
        self.base_algo.fit(X_data, y_data)
        curr_pred = self.base_algo.predict(X_data)

        for iter_num in range(self.iters):
            resid = antigrad(y_data, curr_pred)
            algo = DecisionTreeRegressor(**self.tree_params_dict, random_state=1)
            algo.fit(X_data, resid)
            self.estimators.append(algo)

            weight_tuner = lambda x: loss_function(y_data, curr_pred + x * algo.predict(X_data))
            res, *_ = fmin_slsqp(weight_tuner, np.array([0.05,]), bounds=[(-1, 1)], iprint=0)

            self.weights.append(res)
            curr_pred += self.tau * algo.predict(X_data)
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator, weight in zip(self.estimators, self.weights):
            res += self.tau * estimator.predict(X_data)
        return np.array([1 if x else -1 for x in res > 0.])
