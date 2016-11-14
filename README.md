Neural nets intro
=================

Leipzig Python User Group Meeting, 2016-11-08, 7PM CEST.

There are some [slides](https://github.com/miku/nntour/blob/master/Slides.pdf) available.

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

To generate a visualization of the perceptron learning algorithm, run:

```shell
$ make perceptron.gif
...
```

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

Why not XOR and only XOR-ish. This is only an example and the underlying
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

The do forward-pass and backward propagation a given number of times, e.g. 10000.

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

