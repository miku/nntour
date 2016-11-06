#!/usr/bin/env python
# coding: utf-8

"""
A simple perceptron.

Create gif:

    $ rm -f perceptron-00* perceptron-final*  && python perceptron.py && sh perceptron-gif.sh
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn

def generate_points(N):
    """ Separable points in the plane. Returns X (Nx3), y (1xN). """

    # A hidden weight vector, which is unknown during learning.
    W = np.random.rand(3)

    X, y = [], []

    for i in range(N):
        x1, x2 = [random.uniform(-1, 1) for i in range(2)]
        x = np.array([1, x1, x2])
        s = np.sign(W.T.dot(x))
        X.append(x)
        y.append(s)

    return X, y

def export(filename, X, y, W, title=""):
    """
    Save data plus hyperplane to filname.
    """
    plt.clf()

    pos = np.array([x[1:] for i, x in enumerate(X) if y[i] == 1])
    neg = np.array([x[1:] for i, x in enumerate(X) if y[i] == -1])

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError('bad luck, only a single class')

    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])

    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

    plt.title(title)
    plt.autoscale(enable=False)

    plt.scatter(pos[:, 0], pos[:, 1], color='b')
    plt.scatter(neg[:, 0], neg[:, 1], color='r')

    xline = np.linspace(-2, 2, 1000)
    yline = (-W[0] - W[1] * xline) / W[2]

    plt.plot(xline, yline, '-', color='k')
    plt.savefig(filename)

def pla(X, y):
    """
    Perceptron learning algorithm.
    """

    W = np.random.rand(3)

    def misclassfied_points(W):
        """ For a given weight vector, return the set of misclassified points. """
        misses = []

        for i, x in enumerate(X):
            s = np.sign(W.T.dot(x))
            # print(i, W)
            if not s == y[i]:
                misses.append((x, y[i]))

        return misses

    iteration = 0
    
    while True:
        misses = misclassfied_points(W)
        print('misses: %d' % len(misses))

        export("perceptron-%08d" % iteration, X, y, W, title='#%s' % iteration)

        if len(misses) == 0:
            break

        point = random.choice(misses)
        W = W + point[1] * point[0]
        iteration += 1

    export("perceptron-final.png", X, y, W, title='#%s FIN' % iteration)

    return W

if __name__ == '__main__':
    X, y = generate_points(100)
    
    pos = np.array([x[1:] for i, x in enumerate(X) if y[i] == 1])
    neg = np.array([x[1:] for i, x in enumerate(X) if y[i] == -1])

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError('bad luck, only a single class')

    # plt.scatter(pos[:, 0], pos[:, 1], color='b')
    # plt.scatter(neg[:, 0], neg[:, 1], color='r')
    # plt.savefig('perceptron-0.png')

    W = pla(X, y)

    # plt.scatter(pos[:, 0], pos[:, 1], color='b')
    # plt.scatter(neg[:, 0], neg[:, 1], color='r')

    # xline = np.linspace(-2, 2, 1000)
    # yline = (-W[0] - W[1] * xline) / W[2]

    # plt.plot(xline, yline, '--')
    # plt.savefig('perceptron-1.png')
