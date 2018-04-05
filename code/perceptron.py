#!/usr/bin/env python
# coding: utf-8

"""
A simple perceptron.

Create gif:

    $ make perceptron.gif
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn
import sys
import tempfile
import os

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

def drawimg(X, y, W, filename=None, title=''):
    """
    Save data plus boundary to filname.
    """
    if not filename:
        _, filename = tempfile.mkstemp(prefix='nntour-')

    plt.clf()

    pos = np.array([x[1:] for i, x in enumerate(X) if y[i] == 1])
    neg = np.array([x[1:] for i, x in enumerate(X) if y[i] == -1])

    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])

    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

    plt.title(title)
    plt.autoscale(enable=False)

    plt.scatter(pos[:, 0], pos[:, 1], color='b')
    plt.scatter(neg[:, 0], neg[:, 1], color='r')

    xb = np.linspace(-2, 2, 1000)
    yb = (-W[0] - W[1] * xb) / W[2]

    plt.plot(xb, yb, '-', color='k')
    plt.savefig(filename)

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

        # draw current state
        filename = "images/perceptron-%08d.png" % iteration
        title = '#%d' % iteration
        drawimg(X, y, W, filename=filename, title=title)

        # core idea: weight update
        point = random.choice(misses)
        W = W + point[1] * point[0]

        iteration += 1

    # draw final state
    filename = 'images/perceptron-END.png'
    title = '#%d END' % iteration
    drawimg(X, y, W, filename=filename, title=title)

    return W

if __name__ == '__main__':
    # path to save images
    if not os.path.exists('images'):
        os.makedirs('images')

    while True:
        # generate example data
        X, y = generate_points(100)

        # check, if we have example data for both classes
        if len(set(y)) > 1:
            break

    W = perceptron_learning_algorithm(X, y)

    print('PLA: final weights: %s' % W)
