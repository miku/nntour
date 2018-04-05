#!/usr/bin/env python
# coding: utf-8

"""
Example with keras.

Suitable card to go with larger models[1]: https://www.amazon.de/dp/B00TWFEIWA

> Using the GPU, I’ll show that we can train deep belief networks up to 15x
> faster than using just the CPU, cutting training time down from hours to
> minutes. [2]

Impressive growth: http://www.nvidia.com/content/events/geoInt2015/LBrown_DL.pdf#page=9


[1] https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
[2] https://blog.dominodatalab.com/gpu-computing-and-deep-learning/
"""

from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils

# output classes
nb_classes = 10

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize input
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(10, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

# > Return the cross-entropy between an approximating distribution and a true
# > distribution. The cross entropy between two probability distributions
# > measures the average number of bits needed to identify an event from a set
# > of possibilities, if a coding scheme is used based on a given probability
# > distribution q, rather than the “true” distribution p
#
# More loss functions: https://keras.io/objectives/#available-objectives

# > Four ways to speed up mini-­‐batch learning
# > [...] rmsprop, http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
# https://keras.io/losses/
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# > Trains the model for a fixed number of epochs.
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# $ python hellokeras.py
# Using TensorFlow backend.
# ____________________________________________________________________________________________________
# Layer (type)                     Output Shape          Param #     Connected to
# ====================================================================================================
# dense_1 (Dense)                  (None, 512)           401920      dense_input_1[0][0]
# ____________________________________________________________________________________________________
# activation_1 (Activation)        (None, 512)           0           dense_1[0][0]
# ____________________________________________________________________________________________________
# dense_2 (Dense)                  (None, 10)            5130        activation_1[0][0]
# ____________________________________________________________________________________________________
# activation_2 (Activation)        (None, 10)            0           dense_2[0][0]
# ====================================================================================================
# Total params: 407050
# ____________________________________________________________________________________________________
# Epoch 1/5
# 60000/60000 [==============================] - 16s - loss: 0.1997 - acc: 0.9406
# Epoch 2/5
# 60000/60000 [==============================] - 16s - loss: 0.0919 - acc: 0.9733
# Epoch 3/5
# 60000/60000 [==============================] - 16s - loss: 0.0663 - acc: 0.9815
# Epoch 4/5
# 60000/60000 [==============================] - 16s - loss: 0.0536 - acc: 0.9854
# Epoch 5/5
# 60000/60000 [==============================] - 17s - loss: 0.0425 - acc: 0.9888
# ('Test score:', 0.092632722640130671)
# ('Test accuracy:', 0.97799999999999998)
