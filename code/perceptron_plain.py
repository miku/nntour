#!/usr/bin/env python
# coding: utf-8

"""
A simple perceptron. Plain version.
"""

import numpy as np
import random
import sys

def generate_points(N, dim=2, xmin=-1, xmax=1):
    """
    Separable points dim dimensions. Returns X (N x dim + 1), y (1 x N).
    """
    W, X, y = np.random.rand(dim + 1), [], []

    for i in range(N):
        r = [random.uniform(xmin, xmax) for i in range(dim)]
        x = np.append([1], np.array(r))
        s = np.sign(W.T.dot(x))
        X.append(x)
        y.append(s)

    return X, y

def generate_non_separable_points(N, p=0.0):
    """
    Generate non separable points, random version. Flip separable label with
    probability p.
    """
    X, y = generate_points(N)
    yy = []
    for v in y:
        if random.random() < p:
            yy.append(-1 * v)
        else:
            yy.append(v)
    return X, yy

def perceptron_learning_algorithm(X, y, directory='images'):
    """
    Perceptron learning algorithm.
    """

    # initialize weights
    W = np.random.rand(3)

    def misclassfied_points(W):
        """
        For a given weight vector, return a list of misclassified points.
        """
        misses = []

        for i, x in enumerate(X):
            if np.sign(W.T.dot(x)) != y[i]:
                misses.append((x, y[i]))

        return misses

    # count number of iterations
    iteration = 0
    
    while True:
        misses = misclassfied_points(W)
        print('PLA %s, misses: %d' % (W, len(misses)), file=sys.stderr)

        # all examples classified correctly
        if len(misses) == 0:
            break

        # weight update
        point = random.choice(misses)
        W = W + point[1] * point[0]

        iteration += 1

    return W

if __name__ == '__main__':
    while True:
        # generate example data
        X, y = generate_points(100)

        # check, if we have example data for both classes
        if len(set(y)) > 1:
            break

    W = perceptron_learning_algorithm(X, y)

    print('PLA: final weights: %s' % W)
