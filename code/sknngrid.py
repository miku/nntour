#!/usr/bin/env python
# coding: utf-8

"""
Grid search with NN.

Fitting 3 folds for each of 16 candidates, totalling 48 fits on an 8-core
Xeon(R) CPU E5-2609 0 @ 2.40GHz takes about 24 minutes.

Dataset is MNIST, the Drosophila of machine learning.
"""

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import json
import multiprocessing
import numpy as np
import pandas as pd

class SeqEncoder(json.JSONEncoder):
    """
    Helper to encode additional types.
    
    > To use a custom `JSONDecoder` subclass, specify it with the `cls` kwarg;
    > otherwise `JSONDecoder` is used. https://is.gd/X1GZXX
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
        'hidden_layer_sizes': ((1,), (2,), (5,), (2, 2), (10,), (50,), (50, 50), (100,)),
        'activation': ('relu', 'tanh'),

        # 'hidden_layer_sizes': ((1,)),
        # 'activation': ('relu',),
    }

    mlp = MLPClassifier()
    clf = GridSearchCV(mlp, parameters, verbose=10, n_jobs=multiprocessing.cpu_count(), cv=3)

    # run parameter search
    clf.fit(X_train, y_train)

    print(json.dumps(clf.cv_results_, cls=SeqEncoder))

    # df = pd.DataFrame(clf.cv_results_) # df = pd.read_json("sknngrid.json")
    # cols = [c for c in df.columns if 'param_' in c] + ["mean_test_score"]
    # print(df[cols].sort_values(by="mean_test_score"))

#    param_activation param_hidden_layer_sizes  mean_test_score
# 8              tanh                      [1]         0.377567
# 0              relu                      [1]         0.425683
# 3              relu                   [2, 2]         0.629633
# 11             tanh                   [2, 2]         0.670550
# 9              tanh                      [2]         0.672917
# 1              relu                      [2]         0.682650
# 10             tanh                      [5]         0.886200
# 2              relu                      [5]         0.892267
# 12             tanh                     [10]         0.923050
# 4              relu                     [10]         0.929633
# 14             tanh                 [50, 50]         0.962433
# 13             tanh                     [50]         0.962500
# 6              relu                 [50, 50]         0.964950
# 5              relu                     [50]         0.965833
# 15             tanh                    [100]         0.970833
# 7              relu                    [100]         0.972000
