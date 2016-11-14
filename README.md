Neural nets intro
=================

[Leipzig Python User Group](https://twitter.com/lpyug) Meeting, 2016-11-08, 7PM CEST.

There are [slides](https://github.com/miku/nntour/blob/master/Slides.pdf) available.

Code examples
-------------

Code examples go progressively from a simple perceptron to multi-layer
perceptron to basic examples in tensorflow and keras.

If you want to re-generate the gifs, you'll additionally need:

* [convert](https://www.imagemagick.org/script/convert.php)
* [gifsicle](https://www.lcdf.org/gifsicle/)

Setup a virtual environment:

```shell
$ git clone git@github.com:miku/nntour.git
$ cd nntour
$ mkvirtualenv nntour
$ pip install -r requirements.txt
```

Perceptron
----------

Back in the 60', the perceptron had to be wired together:

![](https://raw.githubusercontent.com/miku/nntour/master/images/mark_i_perceptron.jpg?token=AADRyZFbva3GpVXGje6ozLbLEJ2c6R_zks5YM1s3wA%3D%3D)

Writing python is much more convenient.

```shell
$ cd code
$ python perceptron.py
PLA [ 0.41936044  0.63425651  0.61163369], misses: 10
PLA [ 1.41936044 -0.02270848  0.38862569], misses: 13
PLA [ 0.41936044  0.32147436  1.02536083], misses: 23
...
PLA [ 3.41936044  3.62861005  3.65291291], misses: 1
PLA [ 4.41936044  2.852118    3.45032823], misses: 6
PLA [ 3.41936044  3.14911762  4.09151075], misses: 0
PLA: final weights: [ 3.41936044  3.14911762  4.09151075]
```

To generate a visualization of the perceptron learning algorithm (PLA), run:

```shell
$ make perceptron.gif
...
```

It learns a perfect boundary on separable data in finite steps:

![](https://raw.githubusercontent.com/miku/nntour/master/gifs/perceptron-pla-14-steps.gif?token=AADRybgfQ0WmVaU-NZbgwHdoFhCN-XdVks5YMzirwA%3D%3D)

Random weights
--------------

Just as an illustration, we could just update weights randomly. We do 30
iterations and keep the weights, that resulted in a lowest number of
misclassifations.

```shell
$ make random.gif
Random [ 0.43490987  0.87102943  0.86271172], misses: 2
Random [ 0.06800412  0.48192075  0.23343312], misses: 18
Random [ 0.18258518  0.49428513  0.63182014], misses: 12
...
Random [ 0.66580646  0.60613739  0.1751008 ], misses: 19
Random [ 0.09629388  0.11127787  0.40534235], misses: 17
Random: final weights: [ 0.60895934  0.61896042  0.39653134]
```

![](https://raw.githubusercontent.com/miku/nntour/master/gifs/random-weight-updates-12-misses-30-steps.gif?token=AADRycwXZArATTxIvSy-FbFoUP69glGIks5YMznTwA%3D%3D)

Not too bad, but this data set is also quite simple. Random weight would not
work well in more complated settings.

Pocket algorithm
----------------

Variation of the perceptron learning algorithm, that works with non-separable
data too. Use the PLA for separable data, but remember the weights, that led
to the lowest number of misclassifations.

We iterate 50 times, the draw the best weight found so far.

```shell
$ make pocket.gif
pocket [ 0.26545497  0.68834657  0.85203842], misses: 18
pocket [ 1.26545497 -0.13309076  0.45653466], misses: 32
pocket [ 0.26545497 -0.02342423  1.12487766], misses: 28
...
pocket [ 1.26545497  1.09307412  1.61863759], misses: 25
pocket [ 0.26545497  0.71145003  2.3344745 ], misses: 22
pocket: final weights: [ 0.26545497  0.68834657  0.85203842]
```

![](https://raw.githubusercontent.com/miku/nntour/master/gifs/pocket-0.1-noise-50-steps.gif?token=AADRyY4AHyWd4J3ptro3MS8d4Qo7uWElks5YMzrlwA%3D%3D)

A line cannot separate XOR
--------------------------

The XOR function cannot be separated by a line. The perceptron learning algorithm,
albeit simple and powerful is not able to learn good weights:

```shell
$ make xorish.gif
xorish [ 0.83250694  0.46229229  0.65215989], misses: 52
xorish [-0.16749306  1.13984365 -0.07202366], misses: 50
xorish [ 0.83250694  0.83194263 -0.68367205], misses: 38
...
xorish [-0.16749306  0.92461352  0.09322218], misses: 47
xorish [ 0.83250694  0.14551118 -0.16669564], misses: 50
xorish: final weights: [ 0.83250694  1.82534411 -2.09731963]
```

Example with 50 iterations. 

![](https://raw.githubusercontent.com/miku/nntour/master/gifs/xorish-50-steps.gif?token=AADRyfLyGFooSEFwxhYeV1keU9C1Toaeks5YM1zwwA%3D%3D)

A basic neural network
----------------------

The perceptron is a limited model. A neural net combines multiple perceptrons
in one or more layers.

The example neural network has a single hidden layer with four nodes. Input is
propageted into the hidden layer, then, from the hidden layer to the output
layer. Essentially we take the *weighted sum* of the inputs into a node and pass
it though an *activation function*.

The architecture of out example looks like this:

![](https://raw.githubusercontent.com/miku/nntour/master/code/basicnn.png?token=AADRyUWCnPZsiXLrdKs9NDpR7ulSWR5kks5YMz0HwA%3D%3D)

We have three input values (1x3), which we map with weights (3x4) into four
nodes in the hidden layer (1x4). We then map the values from the hidden layer
(1x4) with another set of weights (4x1) into the output layer, which will
result in a single number.

In our example we want to learn an XOR-ish boolean function.

```python
    X = np.array([
        [0, 0, 1],  # 0
        [0, 1, 1],  # 1
        [1, 0, 1],  # 1
        [1, 1, 1]]) # 0
    y = np.array([[0, 1, 1, 0]]).T
```

Why not XOR and only XOR-ish? This is only an example and the underlying
process could be something else than XOR (over the first two columns). For
example, it is also true, that the output is zero, if we have an odd number
of 1s in the input.

This is only very little data to learn from. More examples almost always yield
better models.

We initialize the weights:

```python
    # 4-node hidden layer
    s0 = np.random.random((3, 4))
    # output layer
    s1 = np.random.random((4, 1))
```

We do a forward-pass, compute a loss (measure of our *wrongness*) and [backprop](https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf)
the error a fixed number of iterations, e.g. 10000.

```python
    # 10000 x FP, BP
    for j in range(10000):
        # sigmoid activation
        l1 = 1 / (1 + np.exp(-np.dot( X, s0)))
        l2 = 1 / (1 + np.exp(-np.dot(l1, s1)))

        # loss function
        l2_delta = (y - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(s1.T) * (l1 * (1 - l1))

        # weight update
        s1 += l1.T.dot(l2_delta)
        s0 += X.T.dot(l1_delta)
```

Finally we can test our model on unseen data. Since this is a boolean
function, we can actually enumerate the whole domain:

```python
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
```

One nice property of neural nets is the asymmetry between training and
testing. It can take weeks to train a complicated neural network, but
milliseconds to use it, once we have some weights.

For the state-of-art [ImageNet](http://image-net.org/) competitions, you can
find weight files on the [internet](http://pjreddie.com/darknet/imagenet/).

We compute the activations and pretty print the results:

```python
    # activations for all examples at once
    l1 = 1 / (1 + np.exp(-np.dot(test_data, s0)))
    l2 = 1 / (1 + np.exp(-np.dot(l1, s1)))

    table = tabulate(np.concatenate((test_data.astype(np.int), np.around(l2, decimals=2)), axis=1),
                     headers=['x', 'x', 'x', 'yhat'], tablefmt='simple')
    print(table)
```

The output tells us, what the neural net computes as output (yhat) for a given
input (x).

```shell
$ python basicnn.py

  x    x    x    yhat
---  ---  ---  ------
  0    0    0    0.11
  0    0    1    0
  0    1    0    0.97
  0    1    1    0.99
  1    0    0    0.97
  1    0    1    0.99
  1    1    0    0.02
  1    1    1    0.02
```

Here, it learned XOR over the first two columns perfectly. But since neural
nets are probabilistic (the weights are initialized randomly and it only saw a
very limited number of examples), it can yield other results as well.

```shell
$ python basicnn.py

 x    x    x    yhat
---  ---  ---  ------
  0    0    0    0.05
  0    0    1    0
  0    1    0    0.51
  0    1    1    0.99
  1    0    0    0.52
  1    0    1    0.99
  1    1    0    0.01
  1    1    1    0.01
```

Here, it is still on the XOR side (since 0.51 and 0.52 can be rounded to 1),
but it is less certain in the case 0-1-0 and 1-0-0.

```shell
  x    x    x    yhat
---  ---  ---  ------
  0    0    0    0.05
  0    0    1    0.01
  0    1    0    0.16
  0    1    1    0.99
  1    0    0    0.57
  1    0    1    0.99
  1    1    0    0
  1    1    1    0.01
```

Here, 0-1-0 is misclassfied (if we assume XOR) as 0 (0.16 rounded down).

The complete code, since it's small.

```python
#!/usr/bin/env python
# coding: utf-8

"""
Basic network.
"""

import numpy as np
from tabulate import tabulate

if __name__ == '__main__':
    X = np.array([
        [0, 0, 1],  # 0
        [0, 1, 1],  # 1
        [1, 0, 1],  # 1
        [1, 1, 1]]) # 0
    y = np.array([[0, 1, 1, 0]]).T

    # 4-node hidden layer
    s0 = np.random.random((3, 4))
    # output layer
    s1 = np.random.random((4, 1))

    # 10000 x FP, BP
    # TODO: numba.jit
    for j in range(10000):
        # sigmoid activation
        l1 = 1 / (1 + np.exp(-np.dot( X, s0)))
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

    table = tabulate(np.concatenate((test_data.astype(np.int), np.around(l2, decimals=2)), axis=1),
                     headers=['x', 'x', 'x', 'yhat'], tablefmt='simple')
    print(table)

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
```

Multi-layer perceptron with scikit-learn
----------------------------------------

Before we go on, we introducte the MNIST dataset. Handwritten digits. The
drosophyla of machine learning tasks.

Here are 32 samples of input images (28x28) from MNIST:

![](https://raw.githubusercontent.com/miku/nntour/master/code/mnistimage.png?token=AADRyVdLR1KdvKbLRFN8F-rRhXMBGjKFks5YM1c3wA%3D%3D)


With scikit-learn, setting up an architecture is done in the constructor. Here
we use two hidden layers with 100 nodes each and stochastic gradient descent:

```python
    mlp = MLPClassifier(verbose=10,
                        hidden_layer_sizes=(100, 100),
                        max_iter=400, alpha=1e-4,
                        solver='sgd',
                        tol=1e-4,
                        random_state=1,
                        learning_rate_init=.1)
```

The model is then fitted to the training data and evaluated on the test set.
It's very compact, so the complete code fits here as well:

```python
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
```

When you run it, you can see the weights converging:

```shell
$ python hellosklearn.py
Iteration 1, loss = 0.28833772
Iteration 2, loss = 0.10794344
...
Iteration 20, loss = 0.00887355
Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
training set score: 0.998233
test set score: 0.978500
```

The model achieved a test score of 97.8, which is similar to the performance
of a human on this task.

Optimizing hyperparameters
--------------------------

We use the training data to learn the weights of a given neural network
architecture. We can also learn the parameters of the architecute, if we split
up our data carefully.

To find a good architecture, we can use GridSearchCV from the scikit-learn library. We
define our parameters like a grid and the grid searcher will run each model and report
the results.

```python
    parameters = {
        'hidden_layer_sizes': ((1,), (2,), (5,), (2, 2), (10,), (50,), (50, 50), (100,)),
        'activation': ('relu', 'tanh'),
    }

    mlp = MLPClassifier()
    clf = GridSearchCV(mlp, parameters, verbose=10, n_jobs=multiprocessing.cpu_count(), cv=3)
```

The complete code can be found in [sknngrid.py](https://github.com/miku/nntour/blob/master/code/sknngrid.py).
Since we have to evaluate a lot of models, this can actually take some time:

```shell
$ python sknngrid.py
```

JSON-serialized results of such a search can be found in this [file](https://raw.githubusercontent.com/miku/nntour/master/code/sknngrid.json?token=AADRye08kPwWDf3DdCdxs58kuEoXYnprks5YM1qCwA%3D%3D).

You can read this quickly with Pandas:

```python
>>> df = pd.read_json("sknngrid.json")
>>> cols = [c for c in df.columns if 'param_' in c] + ["mean_test_score"]
>>> print(df[cols].sort_values(by="mean_test_score"))

#    param_activation param_hidden_layer_sizes  mean_test_score
# 8              tanh                      [1]         0.377567
# 0              relu                      [1]         0.425683
# 3              relu                   [2, 2]         0.629633
# 11             tanh                   [2, 2]         0.670550
# 9              tanh                      [2]         0.672917
# 1              relu                      [2]         0.682650
# 10             tanh                      [5]         0.886200
# 2              relu                      [5]         0.892267
# 12             tanh                     [10]         0.923050
# 4              relu                     [10]         0.929633
# 14             tanh                 [50, 50]         0.962433
# 13             tanh                     [50]         0.962500
# 6              relu                 [50, 50]         0.964950
# 5              relu                     [50]         0.965833
# 15             tanh                    [100]         0.970833
# 7              relu                    [100]         0.972000
```

Recent libraries
----------------

Tensorflow and keras are newer libraries for building neural networks. They
work better with GPUs.

* [hellotf.py](https://github.com/miku/nntour/blob/master/code/hellotf.py)
* [hellokeras.py](https://github.com/miku/nntour/blob/master/code/hellokeras.py)

Cast of [running the examples](https://asciinema.org/a/6x8kv2b7x4ba5rw1yjnrt3yqk?autoplay=1).

![](https://github.com/miku/nntour/raw/master/gifs/tfkeras.gif)

Appendix
--------

Activation function options:

![](https://raw.githubusercontent.com/miku/nntour/master/code/sigmoid_fncs.png?token=AADRya7DenPlrhsD2124SUoRe4INHHwVks5YM1upwA%3D%3D)
