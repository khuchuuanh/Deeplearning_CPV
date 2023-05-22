import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return  tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, stride = [1, stride, stride, 1], padding = 'VALID')

x = tf.placeholder(tf.float32, shape = [None, 66,200,3])
y_ = tf.placeholder(tf.float32, shape = [None, 1])

x_image = x

# first convolutional layer

W_conv1 = weight_variable([5,5,3,24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)

# second convolutional layer
W_conv2 = weight_variable([5,5,3,24])
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

# third convolutional layer
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

# fourth convolutional layer

W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

#fifth convolutional layer
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

