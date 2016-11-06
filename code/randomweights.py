#!/usr/bin/env python
# coding: utf-8

"""
Demonstration of the vast solution space.

What would happen, if we just randomly set the weights?

Do we ever get a correct classification?
"""
from __future__ import print_function
import numpy as np
import sys
import os

from perceptron import generate_points, drawimg

def random_guesses(X, y, directory='images', max_iterations=30):
    """
    Just make a guess. An informed weight "update" does not happen at all.
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

    # keep not of best weights and number of missed points
    bestW, bestMisses = W, len(X)
    
    while True:
        misses = misclassfied_points(W)
        print('Random %s, misses: %d' % (W, len(misses)), file=sys.stderr)

        # all examples classified correctly
        if len(misses) == 0 or iteration == max_iterations:
            break

        # if these weight are better, take note
        if len(misses) < bestMisses:
            bestMisses = len(misses)
            bestW = W

        # draw current state
        filename = "images/random-%08d" % iteration
        title = '#%d' % iteration
        drawimg(X, y, W, filename=filename, title=title)

        # next guess
        W = np.random.rand(3)

        iteration += 1

    # use best W
    W = bestW

    # draw final state
    filename = 'images/random-END'
    title = '#%d END' % iteration
    drawimg(X, y, W, filename=filename, title=title)

    return W

if __name__ == '__main__':
    # path to save images
    if not os.path.exists('images'):
        os.makedirs('images')

    # generate example data
    X, y = generate_points(100)

    # check, if we have example data for both classes
    if len(set(y)) == 1:
        raise ValueError('bad luck, sample data has only a single class')

    W = random_guesses(X, y)

    print('Random: final weights: %s' % W)
