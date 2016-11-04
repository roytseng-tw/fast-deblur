import tensorflow as tf
from util import layer

def net_style(image):
    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(conv2d(image, 3, 32, 9, 1))
    with tf.variable_scope('conv2'):
        conv2 = tf.nn.relu(conv2d(conv1, 32, 64, 3, 2))
    with tf.variable_scope('conv3'):
        conv3 = tf.nn.relu(conv2d(conv2, 64, 128, 3, 2))
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1 = tf.nn.relu(conv2d_transpose(res5, 128, 64, 3, 2))
    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(conv2d_transpose(deconv1, 64, 32, 3, 2))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(conv2d_transpose(deconv2, 32, 3, 9, 1))
    return deconv3 * 127.5

def net_sr_x4(image):
    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(conv2d(image, 3, 64, 9, 1))
    with tf.variable_scope('res1'):
        res1 = residual(conv1, 64, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 64, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 64, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 64, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1 = tf.nn.relu(conv2d_transpose(res4, 64, 64, 3, 2))
    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(conv2d_transpose(deconv1, 64, 64, 3, 2))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(conv2d_transpose(deconv2, 64, 3, 9, 1))
    return deconv3 * 127.5

def net_sr_x8(image):
    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(conv2d(image, 3, 64, 9, 1))
    with tf.variable_scope('res1'):
        res1 = residual(conv1, 64, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 64, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 64, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 64, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1 = tf.nn.relu(conv2d_transpose(res4, 64, 64, 3, 2))
    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(conv2d_transpose(deconv1, 64, 64, 3, 2))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.relu(conv2d_transpose(deconv2, 64, 64, 3, 2))
    with tf.variable_scope('deconv4'):
        deconv4 = tf.nn.tanh(conv2d_transpose(deconv3, 64, 3, 9, 1))
    return deconv4 * 127.5

def net_sr_x4_same(image):
    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(conv2d(image, 3, 64, 9, 1))
    with tf.variable_scope('res1'):
        res1 = residual(conv1, 64, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 64, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 64, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 64, 3, 1)
    with tf.variable_scope('output'):
        output = tf.nn.tanh(conv2d(res4, 64, 3, 9, 1))
    return output * 127.5
