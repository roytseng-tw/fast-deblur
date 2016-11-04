import tensorflow as tf
from util import layer

class SRNet_x4(object):
    """docstring for SRNet_x4"""
    def __init__(self, inputs, reuse=False):
        super(SRNet_x4, self).__init__()
        self.inputs = inputs
        self.output = self.__setup(reuse)
        self.variables = tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES)

    def __setup(self, reuse):
        with tf.variable_scope(self.__class__.__name__, reuse=reuse):
            with tf.variable_scope('conv1'):
                conv1 = tf.nn.relu(layer.conv2d(self.inputs, 3, 64, 9, 1))
            with tf.variable_scope('res1'):
                res1 = layer.residual_noRelu(conv1, 64, 3, 1)
            with tf.variable_scope('res2'):
                res2 = layer.residual_noRelu(res1, 64, 3, 1)
            with tf.variable_scope('res3'):
                res3 = layer.residual_noRelu(res2, 64, 3, 1)
            with tf.variable_scope('res4'):
                res4 = layer.residual_noRelu(res3, 64, 3, 1)
            with tf.variable_scope('deconv1'):
                deconv1 = tf.nn.relu(layer.conv2d_transpose(res4, 64, 64, 3, 2))
            with tf.variable_scope('deconv2'):
                deconv2 = tf.nn.relu(layer.conv2d_transpose(deconv1, 64, 64, 3, 2))
            with tf.variable_scope('output'):
                output = tf.nn.tanh(layer.conv2d(deconv2, 64, 3, 9, 1))
            return output * 127.5

if __name__ == '__main__':
    #g = tf.Graph()
    with tf.get_default_graph().as_default():
        srnet = SRNet_x4(tf.placeholder(tf.float32, shape=(None, 481, 321, 3)))
    print(srnet.__class__.__name__)
    srnet2 = SRNet_x4(tf.placeholder(tf.float32, shape=(None, 300, 200, 3)), 
                      reuse=True)
    for (var1,var2) in zip(srnet.variables,srnet2.variables):
        print(var1 == var2)
