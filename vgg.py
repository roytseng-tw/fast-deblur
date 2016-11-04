# Copied from https://github.com/anishathalye/neural-style
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import misc

def _conv_layer(input, weights, bias, name=None):
    conv = tf.nn.conv2d(input, tf.get_variable('weights', weights.shape, initializer=tf.constant_initializer(weights), trainable=False),
        strides=(1, 1, 1, 1), padding='SAME', name=name)
    return tf.nn.bias_add(conv, tf.get_variable('bias', bias.shape, initializer=tf.constant_initializer(bias), trainable=False))

def _pool_layer(input, name=None):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME', name=name)

def preprocess(image, mean_pixel):
    return image - mean_pixel

def unprocess(image, mean_pixel):
    return image + mean_pixel

# for checking
layer_names = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

model = None
mean_pixel = None
layers = None

def loadmat(model_path):
    global model, mean_pixel, layers
    model = scipy.io.loadmat(model_path, squeeze_me=True, struct_as_record=False)
    mean_pixel = model['meta'].normalization.averageImage
    layers = model['layers'][:35]

def net(input_images, reuse=False):
    net = {}
    current = input_images
    for i, layer in enumerate(layers):
        assert layer.name == layer_names[i]
        with tf.variable_scope(layer.name, reuse=reuse):
            if layer.type == 'conv':
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                weights = layer.weights[0].transpose(1, 0, 2, 3)
                bias = layer.weights[1]
                current = _conv_layer(current, weights, bias, name=layer.name)
            elif layer.type == 'relu':
                current = tf.nn.relu(current, name=layer.name)
            elif layer.type == 'pool':
                current== _pool_layer(current, name=layer.name)
        net[layer.name] = current
    assert len(net) == len(layers)
    return net
