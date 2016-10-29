ML tour
=======

Excerpts from [CS156, Neural Networks](https://www.youtube.com/watch?v=Ih5Mr93E-2c).

Past models: perceptron, linear regression, logistic regression.

Arbitrary non-linear twice-differentiable (smooth) function. Next best: convex
function: gradient descent. Go along the negative of the gradient in fixed
sized steps (learning rates).

More general version: stochastic gradient descent.

Intro
-----

Easy to implement. Not model of choice, rather SVM.

SGD
---

GD minimizes an error function (of the weights). Also in-sample error. A
measure between the targets the hypothesis yields and the real targets,
somehow averaged.

The hypothesis needs to be evaluated at *every* point in the sample, because
we want an average error.

GD = batch GD.

SGD uses just one example at the time.

* pick one example (randomly) at a time
* apply GD on that point (remember perceptron learning algorithm)
* think about average direction you decent along, expected value is the same as in batch version

Much cheaper. Randomization (which can be advantageous).

Cases where randomization helps:

* in NN you have lots of hills and valleys in the error surface
* ESCAPE from the local minimum

![](images/Escape_from_the_local_minimum.jpg)

Neural network model
--------------------

TODO.

Backpropagation
---------------

TODO.

Implementation
==============

* Various minimal variants
* Keras, http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
* Tensorflow
* scikit-learn
