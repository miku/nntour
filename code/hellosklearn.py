#!/usr/bin/env python
# coding: utf-8

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata

if __name__ == '__main__':
    # stores data in ~/scikit_learn_data by default
    mnist = fetch_mldata('MNIST original')
    split = 60000

    X, y = mnist.data / 255., mnist.target

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # mlp = MLPClassifier(verbose=10)

    mlp = MLPClassifier(verbose=10,
                        hidden_layer_sizes=(100, 100),
                        max_iter=400, alpha=1e-4,
                        solver='sgd',
                        tol=1e-4,
                        random_state=1,
                        learning_rate_init=.1)

    mlp.fit(X_train, y_train)

    print("training set score: %0.6f" % mlp.score(X_train, y_train))
    print("test set score: %0.6f" % mlp.score(X_test, y_test))

# MLPClassifier with defaults
# ---------------------------
#
# Iteration 1, loss = 0.42878618
# Iteration 2, loss = 0.20505457
# Iteration 3, loss = 0.15200607
# Iteration 4, loss = 0.12161441
# Iteration 5, loss = 0.10157177
# Iteration 6, loss = 0.08580176
# Iteration 7, loss = 0.07357175
# Iteration 8, loss = 0.06570893
# Iteration 9, loss = 0.05695424
# Iteration 10, loss = 0.05000736
# Iteration 11, loss = 0.04464689
# Iteration 12, loss = 0.04013500
# Iteration 13, loss = 0.03532952
# Iteration 14, loss = 0.03185975
# Iteration 15, loss = 0.02797278
# Iteration 16, loss = 0.02470729
# Iteration 17, loss = 0.02242617
# Iteration 18, loss = 0.02026662
# Iteration 19, loss = 0.01808858
# Iteration 20, loss = 0.01541741
# Iteration 21, loss = 0.01444279
# Iteration 22, loss = 0.01286522
# Iteration 23, loss = 0.01087826
# Iteration 24, loss = 0.01033050
# Iteration 25, loss = 0.00887871
# Iteration 26, loss = 0.00810440
# Iteration 27, loss = 0.00696718
# Iteration 28, loss = 0.00619607
# Iteration 29, loss = 0.00622639
# Iteration 30, loss = 0.00530486
# Iteration 31, loss = 0.00490973
# Iteration 32, loss = 0.00426621
# Iteration 33, loss = 0.00383416
# Iteration 34, loss = 0.00344564
# Iteration 35, loss = 0.00342458
# Iteration 36, loss = 0.00443835
# Iteration 37, loss = 0.00375365
# Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
# training set score: 0.999783
# test set score: 0.978400

# MLPClassifier, 2 hidden layers, 100 units each
# ----------------------------------------------

# Iteration 1, loss = 0.28833772
# Iteration 2, loss = 0.10794344
# Iteration 3, loss = 0.07889505
# Iteration 4, loss = 0.06081647
# Iteration 5, loss = 0.05005079
# Iteration 6, loss = 0.03929399
# Iteration 7, loss = 0.03312722
# Iteration 8, loss = 0.02458840
# Iteration 9, loss = 0.02280704
# Iteration 10, loss = 0.01994174
# Iteration 11, loss = 0.01521781
# Iteration 12, loss = 0.01271448
# Iteration 13, loss = 0.01315993
# Iteration 14, loss = 0.00956860
# Iteration 15, loss = 0.00867025
# Iteration 16, loss = 0.00954227
# Iteration 17, loss = 0.00595672
# Iteration 18, loss = 0.00814455
# Iteration 19, loss = 0.00635555
# Iteration 20, loss = 0.00887355
# Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
# training set score: 0.998233
# test set score: 0.978500
