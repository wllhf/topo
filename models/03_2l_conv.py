from __future__ import division

import os

import numpy as np
import tensorflow as tf

from ..img_toolbox.adjustment import per_image_standardization

PATH = "topo_data/set_03"

EPOCHS = 100
BATCH_SIZE = 100
SPLIT_RATIO = 0.5


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# load
samples = np.load(os.path.join(PATH, "patches.npy"))
labels = np.load(os.path.join(PATH, "labels.npy"))

# stats
n = samples.shape[0]
ntrain = int(n * SPLIT_RATIO)
nclasses = labels.max() + 1
patch_size = samples.shape[1:3]
nchannels = samples.shape[3]
d = np.prod(patch_size) * nchannels
print "Samples:", n, patch_size, nchannels, samples.dtype

# one-hot
temp = np.zeros((labels.shape[0], nclasses), dtype='uint8')
temp[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
labels = temp

# normalize
samples = per_image_standardization(samples)

# shuffle
indices = np.random.permutation(n)
samples, labels = samples[indices, :], labels[indices, :]

# split
train = (samples[:ntrain, :], labels[:ntrain, :])
test = (samples[ntrain:, :], labels[ntrain:, :])

# tensorflow
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], nchannels])
y_ = tf.placeholder(tf.float32, shape=[None, nclasses])

# first layer
W_conv1 = weight_variable([5, 5, nchannels, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
W_fc1 = weight_variable([patch_size[0] // 4 * patch_size[1] // 4 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, patch_size[0] // 4 * patch_size[1] // 4 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout (reduce overfitting)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
W_fc2 = weight_variable([1024, nclasses])
b_fc2 = bias_variable([nclasses])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(EPOCHS):
    idx = i % (ntrain // BATCH_SIZE)
    s, e = idx * BATCH_SIZE, (idx + 1) * BATCH_SIZE
    batch = (train[0][s:e, :], train[1][s:e, :])

    if i % 1 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0].astype('float32'), y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0].astype('float32'), y_: batch[1], keep_prob: 0.5})


# testing
print("test accuracy %g" % accuracy.eval(feed_dict={x: test[0].astype('float32'), y_: test[1], keep_prob: 1.0}))
