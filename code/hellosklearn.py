#!/usr/bin/env python
# coding: utf-8

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    split = 60000

    X, y = MNIST.data / MNIST.data.max(), MNIST.target

    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], y[split:]

    mlp = MLPClassifier(hidden_layer_sizes=(n_units, n_layers),
                        max_iter=n_iterations,
                        alpha=1e-4,
                        solver=solver,
                        ...)
