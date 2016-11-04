import tensorflow as tf
import time
import os
from scipy import misc

from models import SRNet_x4

mean_pixel = [123.68, 116.779, 103.939] # ImageNet average from VGG

tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu4_2", "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
tf.app.flags.DEFINE_float("TV_WEIGHT", 1e-5, "Weight for total variation loss")

tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat", "Path to pretrained vgg model")
tf.app.flags.DEFINE_string("MODEL_PATH", "./model_sr", "Trained model saving dir")

tf.app.flags.DEFINE_integer("IMAGE_SIZE", 128, "Max side size of output image")
tf.app.flags.DEFINE_string("TRAIN_IMAGE_PATH", "/home/roytseng/VisionNAS/EDOF-BSDS", "Path to training data dir")
tf.app.flags.DEFINE_string("TRAIN_IMAGE_FILE", "train_pair_tf.lst", "a file contains lines of path to image data and image label")

tf.app.flags.DEFINE_integer("NUM_ITERATION", 10000, "Number of iteration")
tf.app.flags.DEFINE_integer("NUM_EPOCH", 5, "Number of epoch")
tf.app.flags.DEFINE_float("LEARNING_RATE", 1e-3, "Learning rate")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Number of concurrent images to train on")

tf.app.flags.DEFINE_string("OUTPUT_IMAGE", "out.png", "Transformed image path")
tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")

##------------------
tf.app.flags.DEFINE_integer("num_gpus", 1,"How many number of gpus to use")

FLAGS = tf.app.flags.FLAGS

def test_single_feed():
    image = tf.placeholder(tf.uint8, shape=(481,321,3)) # (H,W,C)
    image_submean = tf.to_float(image) - mean_pixel
    net_in = tf.expand_dims( image_submean, 0)
    net_out = SRNet_x4(net_in / 127.5).output
    image_out = tf.image.encode_jpeg(tf.saturate_cast(tf.squeeze(net_out) + mean_pixel, tf.uint8))

    #what if changing the location of saver init
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'training_checkpoints/fast-deblur-model_10-72000')
        feed_dict = {image: misc.imread('104055_blurred.jpg')}
        out_ = sess.run(image_out, feed_dict=feed_dict)
        with open('104055_deblurred_feed.jpg', 'wb') as f:
             f.write(out_)

#--------------------------
def main(argv=None):
    test_single_feed()

if __name__ == '__main__':
    tf.app.run()
