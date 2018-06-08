#!/usr/bin/env python
# coding: utf-8

"""
A simple perceptron.

Create gif:

    $ make xorish.gif
"""

from __future__ import print_function
from perceptron import generate_points, drawimg

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn
import sys
import tempfile
import collections

from perceptron import drawimg

def generate_xor_dataset(N):
    """
    Hand-craft an xor-ish dataset.
    """
    X, y = [], []

    nq = N // 4

    Range = collections.namedtuple('Range', ['label', 'x', 'y'])

    ranges = [
        Range( 1, (-1, 0), (-1, 0)),
        Range( 1, ( 0, 1), ( 0, 1)),
        Range(-1, (-1, 0), ( 0, 1)),
        Range(-1, ( 0, 1), (-1, 0)),
    ]

    for r in ranges:
        for i in range(nq):
            p = [random.uniform(r.x[0], r.x[1]), random.uniform(r.y[0], r.y[1])]
            x = np.append([1], np.array(p))
            X.append(x)
            y.append(r.label)

    return X, y

def pocket_algorithm(X, y, directory='images', max_iterations=50):
    """
    Perceptron learning algorithm, pocket version.
    Can work with non-separable data.
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
    
    # keep note of best weights and number of missed points
    bestW, bestMisses = W, len(X)

    while True:
        misses = misclassfied_points(W)
        print('xorish %s, misses: %d' % (W, len(misses)), file=sys.stderr)

        # all examples classified correctly
        if len(misses) == 0 or iteration == max_iterations:
            break

        # if these weight are better, take note
        if len(misses) < bestMisses:
            bestMisses = len(misses)
            bestW = W

        # draw current state
        filename = "images/xorish-%08d" % iteration
        title = '#%d' % iteration
        drawimg(X, y, W, filename=filename, title=title)

        # core idea: weight update
        point = random.choice(misses)
        W = W + point[1] * point[0]

        iteration += 1

    # use best W
    W = bestW

    # draw final state
    filename = 'images/xorish-END'
    title = '#%d END' % iteration
    drawimg(X, y, W, filename=filename, title=title)

    return W

if __name__ == '__main__':
    # path to save images
    if not os.path.exists('images'):
        os.makedirs('images')

    # generate example data, XOR final enemy
    X, y = generate_xor_dataset(100)

    # check, if we have example data for both classes
    if len(set(y)) == 1:
        raise ValueError('bad luck, sample data has only a single class')

    W = pocket_algorithm(X, y)

    print('xorish: final weights: %s' % W)
