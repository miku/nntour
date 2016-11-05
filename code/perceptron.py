#!/usr/bin/env python
# coding: utf-8

"""
A simple perceptron.
"""

def dot(a, b):
    """ Dot product. """
    if not len(a) == len(b):
        raise ValueError('vectors must be of same length, got %d and %d' % (len(a), len(b)))
    return sum([a * b for a, b in zip(a, b)])

def sign(x):
    if x >= 0:
        return 1
    return 0

def h(x, w, b):
    """ Hypothesis. """
    return sign(dot(x, w) + b)

def pla(X, y):
    """
    Given a dataset, return model.
    """
    # initialize model
    weights, bias = [0 for _ in X[0]], 0

    for i, x in enumerate(X):
        yhat = h(x, weights, bias)
        for j, w in enumerate(weights):
            weights[j] = weights[j] + (y[i] - yhat) * x[j]

    return weights, bias

if __name__ == '__main__':
    # measurements
    X = [
        [0.5, 1.0, 0.2],
        [0.7, 1.0, 0.3],
        [0.2, 0.5, 0.3],
        [0.4, 0.6, 0.2],
    ]

    # measurement class
    y = [0, 0, 1, 1]

    # weight of the model
    W = [0.1, 0.1, 0.1]
    b = 0.1

    W, b = pla(X, y)
    
    for i, x in enumerate(X):
        yhat = h(x, W, b)
        print(yhat, y[i])
