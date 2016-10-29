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
    """
    Set min and max values and give it some padding.
    """
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

def calculate_loss(model):
    """
    Helper function to evaluate the total loss on the dataset.
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, x):
    """
    Helper function to predict an output (0 or 1)
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    """
    This function learns parameters for the neural network and returns the model.

    - nn_hdim: Number of nodes in the hidden layer
    - num_passes: Number of passes through the training data for gradient descent
    - print_loss: If True, print the loss every 1000 iterations

    """     

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
 
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model)))
     
    return model

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

    num_examples = len(X) # training set size
    nn_input_dim = 2 # input layer dimensionality
    nn_output_dim = 2 # output layer dimensionality
     
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01 # learning rate for gradient descent
    reg_lambda = 0.01 # regularization strength


    # Build a model with a 2-dimensional hidden layer
    model = build_model(2, print_loss=True)
     
    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Decision Boundary for hidden layer size 2")
    plt.savefig('minimal-2.png')


    # Build a model with a 3-dimensional hidden layer
    model = build_model(3, print_loss=True)
     
    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Decision Boundary for hidden layer size 3")
    plt.savefig('minimal-3.png')
