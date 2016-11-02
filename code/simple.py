#!/usr/bin/env python
# coding: utf-8

"""
Simple NN.

Activation and weights.

Input layer activation is the input. Will match dimensions of the input. The
weight matrix describes the weights.

Use forward propagation for computing a value. Backpropagation for adjusting
weights.

Weights are what is learned. Choice of different functions (differentiable).

Sigmoid only: f'(t) = f(t) (1 - f(t)).

> However, usually the weights are much more important than the particular
> function chosen.

"""

import numpy as np

Theta0 = np.random.rand(2, 1)
Theta1 = np.random.rand(1, 1)

a0 = np.array([[1, 0]])

print(Theta0)
print(a0)
a1 = np.dot(Theta0, a0)
print(a1)

# TODO: complete toy implementation.
