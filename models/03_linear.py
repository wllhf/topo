from __future__ import division

import os

import numpy as np
import tensorflow as tf

PATH = "topo_data/set_03"

EPOCHS = 100
BATCH_SIZE = 100
SPLIT_RATIO = 0.5

# load
samples = np.load(os.path.join(PATH, "50_flat_bal_patch.npy"))
labels = np.load(os.path.join(PATH, "50_flat_bal_label.npy"))

# stats
n, d = samples.shape[0], samples.shape[1]
ntrain = int(n * SPLIT_RATIO)
nlabels = labels.max() + 1
print "Samples:", n, d, samples.dtype

# one-hot
temp = np.zeros((labels.shape[0], nlabels), dtype='uint8')
temp[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
labels = temp

# normalize
samples = samples / np.iinfo(samples.dtype).max
mean, std = np.expand_dims(samples.mean(axis=1), axis=1), np.expand_dims(np.std(samples, axis=1), axis=1)
samples = samples - mean
samples = samples / np.maximum(std, 1.0 / d)

# shuffle
indices = np.random.permutation(n)
samples, labels = samples[indices, :], labels[indices, :]

# split
train = (samples[:ntrain, :], labels[:ntrain, :])
test = (samples[ntrain:, :], labels[ntrain:, :])

# tensorflow graph
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, d])
y_ = tf.placeholder(tf.float32, shape=[None, nlabels])

W = tf.Variable(tf.zeros([d, nlabels]))
b = tf.Variable(tf.zeros([nlabels]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b

# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(EPOCHS):
    i = i % (ntrain // BATCH_SIZE)
    s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
    batch = (train[0][s:e, :], train[1][s:e, :])

    if i % 1 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0].astype('float32'), y_: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0].astype('float32'), y_: batch[1]})

# testing
print(accuracy.eval(feed_dict={x: test[0].astype('float32'), y_: test[1]}))
