import tensorflow as tf

def conv2d(x, in_channels, out_channels, kernel_size, stride, padding='SAME'):
    with tf.variable_scope('conv2d') as scope:
        weight = tf.get_variable('weight',
            [kernel_size, kernel_size, in_channels, out_channels],
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        output = batch_norm(
            tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME'),
            out_channels)
        return output

def conv2d_transpose(x, in_channels, out_channels, kernel_size, stride, padding='SAME'):
    with tf.variable_scope('conv2d_transpose') as scope:
        weight = tf.get_variable('weight',
            [kernel_size, kernel_size, out_channels, in_channels],
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        input_shape = tf.shape(x)
        batch = input_shape[0]
        height = input_shape[1] * stride
        width = input_shape[2] * stride
        output_shape = [batch, height, width, out_channels]
        output = batch_norm(
            tf.nn.conv2d_transpose(x, weight, output_shape,
                strides=[1, stride, stride, 1], padding='SAME'),
            out_channels)
        return output

def batch_norm(x, out_channels):
    with tf.variable_scope('batch_norm') as scope:
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], keep_dims=True)
        offset = tf.get_variable('offset', 
            initializer=tf.zeros_initializer([out_channels])) # beta
        scale = tf.get_variable('scale', 
            initializer=tf.ones_initializer([out_channels])) # gamma
        epsilon = 1e-5
        output = tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
        return output

def residual_noRelu(x, channels_size, kernel_size, stride, padding='SAME'):
    with tf.variable_scope('residual_noRelu') as scope:
        with tf.variable_scope('conv1'):
            conv1 = conv2d(x , channels_size, channels_size, kernel_size, stride, padding)
        with tf.variable_scope('conv2'):
            conv2 = conv2d(tf.nn.relu(conv1), channels_size, channels_size, kernel_size, stride, padding)
        output = conv2 + x
        return output