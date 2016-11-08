#!/usr/bin/env python
# coding: utf-8

"""
Display input data.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

if __name__ == '__main__':
    # stores data in ~/scikit_learn_data by default
    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data, mnist.target

    fig = plt.figure()

    p = np.random.permutation(len(X))

    # render first 1024 example
    for i in range(1, 33):
        example = X[p[i]].reshape(28, 28)
        ax = fig.add_subplot(8, 4, i)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.imshow(example, cmap=cm.gray)  

    fig.savefig('mnistimage.png')
