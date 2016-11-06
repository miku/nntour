Martin, Notes
=============

----

5 min.

Perceptron. Early model. Inspiration. Simple implementation. Simple algorithm:
PLA. Pocketing.

DONE:

* simple PLA
* random weights
* pocket PLA

----

5 min.

Show what we can learn with perceptron alone. An show pathological cases, for example
XOR for a non-pocket.

Identify linearity as a problem. Does the sigmoid alone help?

DONE:

* xor failure of PLA, even with pocket
* an NN can deal with XOR

----

15 min.

TODO:

Other implementations:

* plain: numpy + stdlib.
* sklearn (0.18 and later): MLPerceptron.
* tensorflow: Graph model.
* keras: Plug-n-Play.

----

In Total:

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

----

Another gif idea: Show, how the weight are become like templates over the
course of learning the weights.

* like in https://www.youtube.com/watch?v=hAeos2TocJ8?t=40m30s

But only for small grayscale images. Maybe templates for diagonals or simplest
shapes.
