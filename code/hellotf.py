#!/usr/bin/env python
# coding: utf-8

"""
Tensorflow, multilayer perceptron.

Visualize with tensorboard:

    $ tensorboard --logdir tf-summary

More tensorflow: http://delivery.acm.org/10.1145/2960000/2951024/p56-saxena.pdf
"""

from __future__ import print_function
import os
import sys
import tempfile

# download MNIST dataset
pwd = os.path.abspath(os.path.dirname(__file__))

path = os.path.join(pwd, "tf-mnist-data")
print(path, file=sys.stderr)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(path, one_hot=True)

import tensorflow as tf

# The training method itself can have many hyperparameters.
# http://colinraffel.com/wiki/neural_network_hyperparameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# model parameters, architecture
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784 # 28x28 images
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def multilayer_perceptron(x, weights, biases):
    """
    Create model.
    """
    # hidden layer + ReLU
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # hidden layer + ReLU
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# setup variables

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}

pred = multilayer_perceptron(x, weights, biases)

# loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.train.SummaryWriter("tf-summary", sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # optimize, loss value
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print('Epoch: %04d, cost=%0.9f' % (epoch + 1, avg_cost))

    print('optimization done.')

    # test model

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

# From https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py

#     $ time python hellotf.py
#     Epoch: 0001, cost=168.040437403
#     Epoch: 0002, cost=42.922752741
#     Epoch: 0003, cost=26.790772676
#     Epoch: 0004, cost=18.768443520
#     Epoch: 0005, cost=13.713787636
#     Epoch: 0006, cost=10.068921963
#     Epoch: 0007, cost=7.535719216
#     Epoch: 0008, cost=5.620109553
#     Epoch: 0009, cost=4.219670208
#     Epoch: 0010, cost=3.242410429
#     Epoch: 0011, cost=2.453972932
#     Epoch: 0012, cost=1.817244119
#     Epoch: 0013, cost=1.429710297
#     Epoch: 0014, cost=1.092236129
#     Epoch: 0015, cost=0.916826690
#     optimization done.
#     Accuracy:  0.9446

#     real    1m16.055s
#     user    3m0.125s
#     sys     0m9.884s
