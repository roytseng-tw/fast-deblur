import tensorflow as tf
from tensorflow.python.client import device_lib
import time
import os
from scipy import misc

import vgg
import model
from models import SRNet_x4
import reader_my as reader

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

def test_single():
    image_in = tf.expand_dims(reader.get_image('104055_blurred.jpg', 0), 0)
    net_out = model.net_sr_x4_re(image_in / 127.5)

    image_out = tf.image.encode_jpeg(tf.saturate_cast(tf.squeeze(net_out)  + reader.mean_pixel, tf.uint8))
    #image_out = tf.saturate_cast(tf.squeeze(net_out) + reader.mean_pixel, tf.uint8)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model_deblur/fast-deblur-model_*-50000')
        out_ = sess.run(image_out)
        print(im.max(), im.min())
        print(type(out_), len(out_))
        with open('xxx.jpg', 'wb') as f:
             f.write(out_)

def test_dir(dirpath):
    imagePaths = [os.path.join(dirpath, name) for name in os.listdir(dirpath)]
    num_images = len(imagePaths)
    imagePath_queue = tf.train.string_input_producer(imagePaths, num_epochs=1, capacity=200)
    tfreader = tf.WholeFileReader()
    imagePath, image_bytes = tfreader.read(imagePath_queue) #first key is the image filepath
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image_submean = tf.to_float(image) - reader.mean_pixel
    imagePath_batch, image_batch = tf.train.batch([imagePath, image_submean], batch_size=1, num_threads=1, dynamic_pad=True)
    net_out = model.net_sr_x4_re(image_batch / 127.5)
    image_deblurred = tf.image.encode_jpeg(tf.saturate_cast(tf.squeeze(net_out) + reader.mean_pixel, tf.uint8))

    # image output directory
    if not os.path.exists('test_bsds/'):
        os.makedirs('test_bsds/')

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, 'model_deblur/fast-deblur-model_*-1000')

        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            for step in xrange(200):
                if coord.should_stop():
                    break
                # im, impath = sess.run([image_deblurred,imagePath_batch])
                # print(step, os.path.basename(impath[0]))
                # with open('test_bsds/'+ os.path.basename(impath[0]), 'wb') as f:
                #     f.write(im)
                im = sess.run(image_deblurred)
                print(step)
                with open('test_bsds/'+ str(step)+'.jpg', 'wb') as f:
                    f.write(im)
        except Exception as e:
           coord.request_stop(e)
           print('Done -- epoch limit reached')
        finally:
           coord.request_stop()
        coord.join(threads) #wait for threads to finish
        elapsed_time = time.time() - start_time

    print('Cost inference time for {} images: {} secs.'.format(num_images, elapsed_time))
    print('{} secs. per image'.format(elapsed_time/num_images))

#UNOD
def test_single2():
    net_in = tf.placeholder()
    net_out = model.net_sr_x4_re(net_in)
    image_out = tf.image.encode_jpeg(tf.saturate_cast(tf.squeeze(net_out) + reader.mean_pixel, tf.uint8))

    #what if changing the location of saver init
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model_deblur/fast-deblur-model_*-22000')
        feed_dict = {}
        out_ = sess.run(image_out, feed_dict=feed_dict)
        print(im.max(), im.min())
        print(type(out_), len(out_))
        with open('104055_deblurred.jpg', 'wb') as f:
             f.write(out_)

#--------------------------
def get_inputs():
    dirpath = '/home/roytseng/VisionNAS/EDOF-BSDS/test/data'
    imagePaths = [os.path.join(dirpath, name) for name in os.listdir(dirpath)]
    num_images = len(imagePaths)
    imagePath_queue = tf.train.string_input_producer(imagePaths, num_epochs=1)
    tfreader = tf.WholeFileReader()
    imagePath, image_bytes = tfreader.read(imagePath_queue) #first key is the image filepath
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image_submean = tf.to_float(image) - reader.mean_pixel
    image_batch = tf.train.batch([image_submean], batch_size=1, num_threads=4, dynamic_pad=True, capacity=1)
    return image_batch


def test_multi_gpu():
    g = tf.Graph()
    with g.as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        image_outputs = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    net_in = get_inputs()
                    srnet = SRNet_x4(net_in)
                    net_out = srnet.output
                    image_deblurred = tf.image.encode_jpeg(
                        tf.saturate_cast(tf.squeeze(net_out) + reader.mean_pixel, tf.uint8))
                    tf.get_variable_scope().reuse_variables()
                    image_outputs.append(image_deblurred)

    ncreader = tf.train.NewCheckpointReader("training_checkpoints/fast-deblur-model_*-36000")
    var_to_shape_map = ncreader.get_variable_to_shape_map()
    ckpt_varnames = var_to_shape_map.keys()
    new_varnames = [var.name for var in g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
    print(len(ckpt_varnames))
    print(len(new_varnames))

    print(type(ckpt_varnames[0]))
    count = 0
    var_dict = {}
    for name in ckpt_varnames:
        for var in g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if name in var.name:
                var_dict[name] = var
                count += 1
                continue;
    print(count)

    saver = tf.train.Saver(var_dict)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(graph=g, config=config) as sess:
        saver.restore(sess, 'training_checkpoints/fast-deblur-model_*-36000')
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            for step in xrange(200):
                if coord.should_stop():
                    break
                outs_ = sess.run(image_outputs)
                #print(len(outs_))
                print(step, os.path.basename(imagePaths[step]))
                with open('test_bsds/'+ os.path.basename(imagePaths[step]), 'wb') as f:
                    f.write(im)
        except Exception as e:
            coord.request_stop(e)
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads) #wait for threads to finish
        elapsed_time = time.time() - start_time

    print('Cost inference time for {} images: {} secs.'.format(num_images, elapsed_time))
    print('{} secs. per image'.format(elapsed_time/num_images))

def main(argv=None):
    #test_dir('/home/roytseng/VisionNAS/EDOF-BSDS/test/data')
    test_multi_gpu()
    #local_device_protos = device_lib.list_local_devices()
    #for x in local_device_protos:
    #    print(x)

if __name__ == '__main__':
    tf.app.run()
