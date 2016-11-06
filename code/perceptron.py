#!/usr/bin/env python
# coding: utf-8

"""
A simple perceptron.
"""

import matplotlib.pyplot as plt
import random
import numpy as np

def generate_points(N):
    """ Separable points in the plane. """

    # A hidden weight vector, which is unknown during learning.
    W = np.random.rand(3, 1)
    # W = np.array([0.7, 1.0, 0.1])
    print(W)

    X, y = [], []

    for i in range(N):
        x1, x2 = [random.uniform(-1, 1) for i in range(2)]
        x = np.array([1, x1, x2])
        s = np.sign(W.T.dot(x))
        X.append(x)
        y.append(s)

    return X, y

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
    # # measurements
    # X = [
    #     [0.5, 1.0, 0.2],
    #     [0.7, 1.0, 0.3],
    #     [0.2, 0.5, 0.3],
    #     [0.4, 0.6, 0.2],
    # ]

    # # measurement class
    # y = [0, 0, 1, 1]

    # # weight of the model
    # W = [0.1, 0.1, 0.1]
    # b = 0.1

    # W, b = pla(X, y)
    
    # for i, x in enumerate(X):
    #     yhat = h(x, W, b)
    #     print(yhat, y[i])

    X, y = generate_points(100)
    
    pos = np.array([x[1:] for i, x in enumerate(X) if y[i] == 1])
    neg = np.array([x[1:] for i, x in enumerate(X) if y[i] == -1])

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError('bad luck, only a single class')

    plt.scatter(pos[:, 0], pos[:, 1], color='b')
    plt.scatter(neg[:, 0], neg[:, 1], color='r')
    plt.savefig('perceptron-0.png')
