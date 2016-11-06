#!/usr/bin/env python
# coding: utf-8

"""
A simple perceptron.

Create gif:

    $ make pocket.gif
"""

from __future__ import print_function
from perceptron import generate_points, drawimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn
import sys
import tempfile

from perceptron import generate_non_separable_points, drawimg

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
        print('pocket %s, misses: %d' % (W, len(misses)), file=sys.stderr)

        # all examples classified correctly
        if len(misses) == 0 or iteration == max_iterations:
            break

        # if these weight are better, take note
        if len(misses) < bestMisses:
            bestMisses = len(misses)
            bestW = W

        # draw current state
        filename = "images/pocket-%08d" % iteration
        title = '#%d' % iteration
        drawimg(X, y, W, filename=filename, title=title)

        # core idea: weight update
        point = random.choice(misses)
        W = W + point[1] * point[0]

        iteration += 1

    # use best W
    W = bestW

    # draw final state
    filename = 'images/pocket-END'
    title = '#%d END' % iteration
    drawimg(X, y, W, filename=filename, title=title)

    return W

if __name__ == '__main__':
    # path to save images
    if not os.path.exists('images'):
        os.makedirs('images')

    # generate example data, with a bit of noise
    X, y = generate_non_separable_points(100, p=0.1)

    # check, if we have example data for both classes
    if len(set(y)) == 1:
        raise ValueError('bad luck, sample data has only a single class')

    W = pocket_algorithm(X, y)

    print('pocket: final weights: %s' % W)
