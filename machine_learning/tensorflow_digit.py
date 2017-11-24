# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:54:24 2017

@author: J
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
data = read_data_sets('MNIST_data', one_hot=True)

lay_dim = [10]
n_iter = 10000
alpha = 0.01

X = tf.placeholder(tf.float32,shape=[None,784], name='X')
y = tf.placeholder(tf.float32, shape=[None,10], name='y')


#W = tf.Variable(tf.random_normal(shape=[784, lay_dim[0]]), name='W')
#b = tf.Variable(tf.zeros(shape=[lay_dim[0]]), name='b')

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y_hat = tf.nn.softmax(tf.matmul(X, W) + b)

cross_entropy = -tf.reduce_sum(y * tf.log(y_hat))

train_step = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
costs = []
for i in range(n_iter):
    batch = data.train.next_batch(50)
    sess.run(train_step, feed_dict={X: batch[0], y:batch[1]})
    
    cost = sess.run(cross_entropy, feed_dict={X: batch[0], y: batch[1]})
    costs.append(cost)
    
    
    if i % 1000 == 0:
        plt.plot(costs)
        plt.show()



correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print(accuracy.eval(feed_dict={X: data.test.images, y: data.test.labels}))
print(sess.run(accuracy, feed_dict={X: data.test.images, y: data.test.labels}))