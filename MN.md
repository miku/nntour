Martin, Notes
=============

----

5 min.

Perceptron. Early model. Inspiration. Simple implementation. Simple algorithm:
PLA. Pocketing.

TODO:

* python code (stdlib only)
* gif of PLA
* gif of Pocket PLA

----

5 min.

Show what we can learn with perceptron alone. An show pathological cases, for example
XOR for a non-pocket.

Identify linearity as a problem. Does the sigmoid alone help?

TODO:

* gif of perceptron with logistic activation function

----

15 min.

Other implementations:

* plain: numpy + stdlib.
* sklearn (0.18 and later): MLPerceptron.
* tensorflow: Graph model.
* keras: Plug-n-Play.

About 25 min.

Should contain lot (5+) of illustrative gifs. Easier to communicate.

Show en-passant:

* how to generate artificial data sets with sklearn
* a bit of numpy
* traditional (non-end-to-end) learning, namely: feature engineering

The feature engineering maybe with a subset of MNIST or even a smaller dataset,
like in Yaser et. al. (2012), show:

* 16x16 images
* only two classes: 5 and 1
* break down image into two features: intensity and symmetry
