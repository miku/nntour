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

```
$ git clone git@github.com:miku/nntour.git
$ cd nntour
$ mkvirtualenv nntour
$ pip install -r requirements.txt
```

Perceptron
----------

```
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

```
$ make perceptron.gif
...
```

![](https://raw.githubusercontent.com/miku/nntour/master/gifs/perceptron-pla-14-steps.gif?token=AADRybgfQ0WmVaU-NZbgwHdoFhCN-XdVks5YMzirwA%3D%3D)

Random weights
--------------

Just as an illustration, we could just update weights randomly. We do 30
iterations and keep the weights, that resulted in a lowest number of
misclassifations.

```
$ make random.gif
Random [ 0.43490987  0.87102943  0.86271172], misses: 2
Random [ 0.06800412  0.48192075  0.23343312], misses: 18
Random [ 0.18258518  0.49428513  0.63182014], misses: 12
...
Random [ 0.45736977  0.80506182  0.13310176], misses: 15
Random [ 0.66580646  0.60613739  0.1751008 ], misses: 19
Random [ 0.09629388  0.11127787  0.40534235], misses: 17
```

![](https://raw.githubusercontent.com/miku/nntour/master/gifs/random-weight-updates-12-misses-30-steps.gif?token=AADRycwXZArATTxIvSy-FbFoUP69glGIks5YMznTwA%3D%3D)

Pocket algorithm
----------------

Variation of the perceptron learning algorithm, that works with non-separable
data too. Use the PLA for separable data, but remember the weights, that led
to the lowest number of misclassifations.

We iterate 50 times, the draw the best weight found so far.

```
$ make pocket.gif
```

![](https://raw.githubusercontent.com/miku/nntour/master/gifs/pocket-0.1-noise-50-steps.gif?token=AADRyY4AHyWd4J3ptro3MS8d4Qo7uWElks5YMzrlwA%3D%3D)

