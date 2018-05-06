#coding=utf-8

from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 1}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05


def loss_function(y, h):
    return sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / len(y)


def der_loss_function(y, h):
    return (h - y) / (h * (1 - h))


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict)
        self.base_algo.fit(X_data, y_data)
        self.estimators = []
        self.weights = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            grad = der_loss_function(y_data, curr_pred)
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            algo = DecisionTreeRegressor(**self.tree_params_dict)
            algo.fit(X_data, -grad)
            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            # Нужно настроить вес для очередного дерева
            weight_tuner = lambda x: loss_function(y_data, curr_pred + x * algo.predict(X_data))
            # варьируем вес и минимизируем loss function
            res = minimize(weight_tuner, np.array([0.5,]))
            self.weights.append(res.x)
            curr_pred += res.x * algo.predict(X_data)
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator, weight in zip(self.estimators, self.weights):
            res += weight * estimator.predict(X_data)
        return res > 0.
