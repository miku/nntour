#!/usr/bin/env python
# coding: utf-8

"""
Grid search with NN.
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

    # df = pd.DataFrame(clf.cv_results_)
    # cols = [c for c in df.columns if 'param_' in c] + ["mean_test_score"]
    # print(df[cols].sort_values(by="mean_test_score"))

#    param_activation param_hidden_layer_sizes  mean_test_score
# 0              relu                      [1]         0.387100
# 6              tanh                      [1]         0.391717
# 3              relu                   [2, 2]         0.484400
# 1              relu                      [2]         0.566933
# 9              tanh                   [2, 2]         0.624783
# 7              tanh                      [2]         0.664400
# 8              tanh                      [5]         0.880667
# 2              relu                      [5]         0.893133
# 10             tanh                     [10]         0.922533
# 4              relu                     [10]         0.928367
# 5              relu                    [100]         0.970050
# 11             tanh                    [100]         0.971133
