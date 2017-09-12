import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc
import skimage
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Data flow graph
input_dim = 784

def weights(shape):
    # return tf.Variable(tf.zeros[shape],tf.float32)
    return tf.Variable(tf.truncated_normal(shape, mean=0.0,stddev=1.0,dtype=tf.float32))

def bias(shape):
    # return tf.Variable(tf.zeros[shape],tf.float32)
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32))

def mul_activation(a,b,c):
    return tf.sigmoid(tf.matmul(a,b)+ c)

def gaussian_noise(d):
    samples = np.random.normal(0,0.25,input_dim)
    e =  d + samples
    return e

def masking_noise(X,rate):

    X1 = X
    n_samples = int(batch_size)
    n_points = input_dim

    for j in range(n_samples):
        mask = np.random.randint(0, int(n_points), rate)

        for m in mask:
            X1[:, m] = tf.zeros([])

    return X1


encoder_units = 1000
decoder_unit = 1000
n_epochs = 50
batch_size = 4
learning_rate = 0.001

x = tf.placeholder(tf.float32,[None, input_dim])
y = tf.placeholder(tf.float32,[None, input_dim]) #validation

# with tf.name_scope("denoise") as scope:
#     x1 = gaussian_noise(x)
#     x1_reshape = tf.reshape(x1,[-1,28,28,1])
#     tf.summary.image("denoised",x1_reshape,10)

with tf.name_scope("denoise_train") as scope:
    #x2 = masking_noise(x,int(0.25*input_dim))
    x2 = gaussian_noise(x)
    x2_reshape = tf.reshape(x2,[-1,28,28,1])
    tf.summary.image("denoised_train",x2_reshape,10)

with tf.name_scope("denoise_valid") as scope:
    #x2 = masking_noise(x,int(0.25*input_dim))
    ya = gaussian_noise(y)
    ya_reshape = tf.reshape(ya,[-1,28,28,1])
    tf.summary.image("denoised_valid",ya_reshape,10)

#encoder weights

with tf.name_scope("weights_1") as scope:
    w1 = weights([input_dim, encoder_units[0]])
    w1_reshape = tf.reshape(w1,[-1,28,28,1]) # not clear
    tf.summary.image("weight1",w1_reshape,1000)

with tf.name_scope("weights_2") as scope:
    w2 = weights([encoder_units[0], encoder_units[1]])
    w2_reshape = tf.reshape(w2, [-1,100,100, 1])
    tf.summary.image("weight2", w2_reshape, 1000)

with tf.name_scope("weights_3") as scope:
    w3 = weights([encoder_units[1], encoder_units[2]])
    w3_reshape = tf.reshape(w3, [-1,100,100, 1])
    tf.summary.image("weight3", w3_reshape, 1000)

#w4 = weights([encoder_units[2], encoder_units[3]])
#w5 = weights([encoder_units[3], encoder_units[4]])
#w6 = weights([encoder_units[4], encoder_units[5]])

# decoder weight
w_ = weights([decoder_unit,input_dim])

# bias encoder

b1 = bias([encoder_units[0]])
b2 = bias([encoder_units[1]])
b3 = bias([encoder_units[2]])
#b4 = bias([encoder_units[3]])
#b5 = bias([encoder_units[4]])
#b6 = bias([encoder_units[5]])

# bias decoder
b_ = bias([input_dim])

# activations_train
y1 = mul_activation(x2,w1,b1)
y2 = mul_activation(y1,w2,b2)
y3 = mul_activation(y2,w3,b3)

y4 = mul_activation(ya,w1,b1)
y5 = mul_activation(y4,w2,b2)
y6 = mul_activation(y5,w3,b3)

#reconstruction train
with tf.name_scope("resconstruction_train") as scope:
    z = mul_activation(y3, w_, b_)
    z_reshape = tf.reshape(z,[-1,28,28,1])
    tf.summary.image("reconstruct",z_reshape,10)

with tf.name_scope("resconstruction_valid") as scope:
    z1 = mul_activation(y6, w_, b_)
    z1_reshape = tf.reshape(z,[-1,28,28,1])
    tf.summary.image("reconstruct",z_reshape,10)

# Loss and optimizer

with tf.name_scope("Loss_train") as scope:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z , x))
    tf.summary.scalar("loss_train",cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope("Loss_valid") as scope:
    cross_entropy_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z1 , y))
    tf.summary.scalar("loss_valid",cross_entropy_valid)
valid_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_valid)

#Run the init

merged = tf.summary.merge_all()
log_path = "event_files_path"
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter(log_path , sess.graph)

#Training

for i in range(n_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    _,k, c, k1, summary = sess.run([train_step, valid_step, cross_entropy, cross_entropy_valid, merged], feed_dict = {x : batch_xs, y : mnist.validation.images})
    writer.add_summary(summary, i)
    writer.flush()
    #print("Epoch:", '%04d' % (i + 1), "Loss_train:", c)
    print("Epoch:", '%04d' % (i + 1), "Loss_valid:", k1)

