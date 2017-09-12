# 3 layer autoencoder with two encoder units and a decoder unit

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)                      # download the MNIST data and labels

# Data flow graph

N = 784
enc = 500
enc1 = 300
n_epochs = 1000
batch_size = 100

x = tf.placeholder(tf.float32,[None, N])
w1 = tf.Variable(tf.random_normal([N, enc],mean=0.0,stddev=1.0,dtype=tf.float32))	#encoder weights
b1 = tf.Variable(tf.random_normal([enc],mean=0.0,stddev=1.0,dtype=tf.float32))

w2 = tf.Variable(tf.random_normal([enc, enc1],mean=0.0,stddev=1.0,dtype=tf.float32))
b2 = tf.Variable(tf.random_normal([enc1],mean=0.0,stddev=1.0,dtype=tf.float32))

y1 = tf.sigmoid(tf.matmul(x,w1) + b1)
y = tf.sigmoid(tf.matmul(y1,w2) + b2)

w_ = tf.Variable(tf.random_normal([enc1, N],mean=0.0,stddev=1.0,dtype=tf.float32))	#decoder weights
b_ = tf.Variable(tf.random_normal([N],mean=0.0,stddev=1.0,dtype=tf.float32))

z = tf.matmul(y,w_)+b_

# Loss and optimizer

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z , x))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

#Run the init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Training

for i in range(n_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    _, c = sess.run([train_step, cross_entropy], feed_dict = {x : batch_xs})
    print("Epoch:", '%04d' % (i + 1), "Loss:", c)
