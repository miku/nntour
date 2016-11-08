#!/usr/bin/env python
# coding: utf-8

"""
The essence.

TODO: find a way to plot this.
"""

import numpy as np

if __name__ == '__main__':
    X = np.array([
        [0, 0, 1],  # 0
        [0, 1, 1],  # 1
        [1, 0, 1],  # 1
        [1, 1, 1]]) # 0
    y = np.array([[0, 1, 1, 0]]).T

    # 3-node hidden layer
    s0 = np.random.random((3, 4))
    # output layer
    s1 = np.random.random((4, 1))

    for j in range(10000):
        # sigmoid activation
        l1 = 1 / (1 + np.exp(-np.dot(X, s0)))
        l2 = 1 / (1 + np.exp(-np.dot(l1, s1)))

        # loss function
        l2_delta = (y - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(s1.T) * (l1 * (1 - l1))

        # weight update
        s1 += l1.T.dot(l2_delta)
        s0 += X.T.dot(l1_delta)

    # test our model on unseen data
    test_data = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]])

    # activations for all examples at once
    l1 = 1 / (1 + np.exp(-np.dot(test_data, s0)))
    l2 = 1 / (1 + np.exp(-np.dot(l1, s1)))

    print(np.concatenate((test_data.astype(np.int), np.around(l2, decimals=2)), axis=1))

# Neural net can approximate XOR of first and second value.
# There is still uncertainty. The training set is small.

# |-------- X ------|-- y -|
#
# [[ 0.    0.    0.    0.11]
#  [ 0.    0.    1.    0.  ]
#  [ 0.    1.    0.    0.99]
#  [ 0.    1.    1.    0.98]
#  [ 1.    0.    0.    0.99]
#  [ 1.    0.    1.    0.98]
#  [ 1.    1.    0.    0.02]
#  [ 1.    1.    1.    0.02]]
#  