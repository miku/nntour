#!/usr/bin/env python
# coding: utf-8

"""
Grid search with NN.
"""

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import json
import numpy as np

class SeqEncoder(json.JSONEncoder):
    """
    Helper to encode additional types.
    
    > To use a custom JSONDecoder subclass, specify it with the cls kwarg;
    > otherwise JSONDecoder is used. https://is.gd/X1GZXX
    """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    # stores data in ~/scikit_learn_data by default
    mnist = fetch_mldata('MNIST original')
    split = 60000

    X, y = mnist.data / 255., mnist.target

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    parameters = {
        'hidden_layer_sizes': ((1,), (2,), (5,), (2, 2), (10,), (100,)),
        # 'hidden_layer_sizes': ((1,)),
        'activation': ('relu', 'tanh'),
        # 'activation': ('relu',),
    }

    mlp = MLPClassifier()
    clf = GridSearchCV(mlp, parameters, verbose=10, n_jobs=4, cv=3)

    clf.fit(X_train, y_train)

    print(json.dumps(clf.cv_results_, cls=SeqEncoder))

# Fitting 3 folds for each of 6 candidates, totalling 18 fits
# [CV] activation=relu, hidden_layer_sizes=(1,) ........................
# [CV] activation=relu, hidden_layer_sizes=(1,) ........................
# [CV] activation=relu, hidden_layer_sizes=(1,) ........................
# [CV] activation=relu, hidden_layer_sizes=(10,) .......................
# [CV]  activation=relu, hidden_layer_sizes=(1,), score=0.371419 -   0.1s
# [CV] activation=relu, hidden_layer_sizes=(10,) .......................
# [CV]  activation=relu, hidden_layer_sizes=(10,), score=0.927065 -   0.1s
# [CV] activation=relu, hidden_layer_sizes=(10,) .......................
# [CV]  activation=relu, hidden_layer_sizes=(1,), score=0.382357 -   0.1s
# [CV] activation=relu, hidden_layer_sizes=(100,) ......................
# [CV]  activation=relu, hidden_layer_sizes=(1,), score=0.368126 -   0.1s
# [CV] activation=relu, hidden_layer_sizes=(100,) ......................
# [CV]  activation=relu, hidden_layer_sizes=(10,), score=0.927296 -   0.2s
# [Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:  2.4min
# [CV] activation=relu, hidden_layer_sizes=(100,) ......................
# [CV]  activation=relu, hidden_layer_sizes=(10,), score=0.930840 -   0.1s
# [CV] activation=tanh, hidden_layer_sizes=(1,) ........................
# [CV]  activation=relu, hidden_layer_sizes=(100,), score=0.972206 -   0.4s
# [CV] activation=tanh, hidden_layer_sizes=(1,) ........................
# [CV]  activation=relu, hidden_layer_sizes=(100,), score=0.969298 -   0.3s
# [CV] activation=tanh, hidden_layer_sizes=(1,) ........................
# [CV]  activation=relu, hidden_layer_sizes=(100,), score=0.972246 -   0.2s
# [CV] activation=tanh, hidden_layer_sizes=(10,) .......................
# [CV]  activation=tanh, hidden_layer_sizes=(1,), score=0.434263 -   0.1s
# [Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:  5.0min
# [CV] activation=tanh, hidden_layer_sizes=(10,) .......................
# [CV]  activation=tanh, hidden_layer_sizes=(1,), score=0.365218 -   0.0s
# [CV] activation=tanh, hidden_layer_sizes=(10,) .......................
# [CV]  activation=tanh, hidden_layer_sizes=(1,), score=0.415312 -   0.0s
# [CV] activation=tanh, hidden_layer_sizes=(100,) ......................
# [CV]  activation=tanh, hidden_layer_sizes=(10,), score=0.923665 -   0.1s
# [Parallel(n_jobs=4)]: Done  13 out of  18 | elapsed:  6.3min remaining:  2.4min
# [CV] activation=tanh, hidden_layer_sizes=(100,) ......................
# [CV]  activation=tanh, hidden_layer_sizes=(10,), score=0.915696 -   0.1s
# [CV] activation=tanh, hidden_layer_sizes=(100,) ......................
# [CV]  activation=tanh, hidden_layer_sizes=(10,), score=0.921638 -   0.2s
# [Parallel(n_jobs=4)]: Done  15 out of  18 | elapsed:  7.4min remaining:  1.5min
# [CV]  activation=tanh, hidden_layer_sizes=(100,), score=0.972905 -   0.4s
# [CV]  activation=tanh, hidden_layer_sizes=(100,), score=0.969548 -   0.3s
# [CV]  activation=tanh, hidden_layer_sizes=(100,), score=0.969495 -   0.2s
# [Parallel(n_jobs=4)]: Done  18 out of  18 | elapsed:  9.3min finished
