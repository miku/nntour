#!/usr/bin/env python
# coding: utf-8

"""
Simplest NN.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

if __name__ == '__main__':
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.savefig('minimal-0.png')

    # In fact, that's one of the major advantages of Neural Networks. You don't need
    # to worry about feature engineering. The hidden layer of a neural network will
    # learn features for you.

    # Train the logistic rgeression classifier
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.title("Logistic Regression")
    plt.savefig('minimal-1.png')
