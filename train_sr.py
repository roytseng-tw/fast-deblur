import tensorflow as tf
import time
import os
import re

import vgg
from models import SRNet_x4
import reader as reader

tf.app.flags.DEFINE_string(
    "TRAIN_IMAGE_DIR",
    "/home/roytseng/VisionNAS/EDOF-BSDS",
    "Path to the root directory of training data.")
tf.app.flags.DEFINE_string(
    "TRAIN_INPUT_FILE_PATH",
    "/home/roytseng/VisionNAS/EDOF-BSDS/train_pair_tf.lst",
    "Path to the file contains paths of image_data and image_label in each line.")

tf.app.flags.DEFINE_string(
    "CONTENT_LAYERS",
    "relu1_2,relu2_2",
    "Which VGG layer to extract content loss from")
#tf.app.flags.DEFINE_float("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
#tf.app.flags.DEFINE_float("TV_WEIGHT", 1e-5, "Weight for total variation loss")

tf.app.flags.DEFINE_string(
    "VGG_PATH",
    "imagenet-vgg-verydeep-19.mat",
    "Path to pretrained vgg model")
tf.app.flags.DEFINE_string(
    "CHECKPOINT_SAVE_DIR",
    "./training_checkpoints",
    "Training checkpoints saving directory.")
tf.app.flags.DEFINE_string(
    "TEST_SAVE_DIR",
    "./training_tests",
    "Training periodic testing output directory.")

tf.app.flags.DEFINE_integer("NUM_ITERATION", 10000, "Number of iteration")
tf.app.flags.DEFINE_integer("NUM_EPOCH", 5, "Number of epoch")
tf.app.flags.DEFINE_float("LEARNING_RATE", 1e-5, "Learning rate")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 4, "Number of concurrent images to train on")
#----------------------------------------------
tf.app.flags.DEFINE_string("OUTPUT_IMAGE", "out.png", "Transformed image path")
tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
#----------------------------------------------

FLAGS = tf.app.flags.FLAGS

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    '''
    tf.slice(input_, begin, size, name=None)
    If size[i] is -1, all remaining elements in dimension i are included in the slice.
    In other words, this is equivalent to setting:
        size[i] = input.dim_size(i) - begin[i]
    '''
    y = tf.slice(layer, [0,0,0,0], [-1,height-1,-1,-1]) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], [-1,-1,width-1,-1]) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

def gram(layer):
    shape = tf.shape(layer)
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([-1, num_filters]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(size)
    return gram

def train():
    # Check model dir
    if not os.path.exists(FLAGS.CHECKPOINT_SAVE_DIR):
        os.makedirs(FLAGS.CHECKPOINT_SAVE_DIR)
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    # Load VGG19 pretrained weights
    vgg.loadmat(FLAGS.VGG_PATH)

    # Return inputs as tensors:
    # mean subtracted, values are between [-127.5, 127.5]
    data_batch, label_batch = reader.get_inputs_from_file(
        FLAGS.TRAIN_INPUT_FILE_PATH, FLAGS.BATCH_SIZE, FLAGS.NUM_EPOCH, scale_factor=0.25)

    # input:  ranged between [-1, 1]
    # output: ranged between [-127.5, 127.5]
    srnet = SRNet_x4(data_batch / 127.5)
    # Compute loss
    # vgg take input range [-127.5, 127.5]
    vgg_deblurred = vgg.net(srnet.output, reuse=False)
    vgg_label = vgg.net(label_batch, reuse=True)

    content_loss = 0.0
    for layer in content_layers:
        deblurred_fmap = vgg_deblurred[layer]
        label_fmap     = vgg_label[layer]
        size = tf.to_float( tf.size(label_fmap))
        content_loss += 2 * tf.nn.l2_loss(deblurred_fmap - label_fmap) / size

    # pixel_loss =  tf.nn.l2_loss(generated - label_batch) * 2 / tf.to_float(tf.size(label_batch))
    total_loss = 10 * content_loss + total_variation_loss(srnet.output)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE).minimize(total_loss, global_step=global_step)

    # For periodic testing
    if not os.path.exists(FLAGS.TEST_SAVE_DIR):
        os.makedirs(FLAGS.TEST_SAVE_DIR)

    image_lr = tf.expand_dims(reader.get_image('104055_blurred.jpg'), 0)
    srnet_test = SRNet_x4( image_lr/127.5, reuse=True)
    image_hr = srnet_test.output
    image_output = tf.image.encode_jpeg(tf.saturate_cast(tf.squeeze(image_hr) + reader.mean_pixel, tf.uint8))

    # maintain epoch counter
    epoch_cnt = None
    total_num_images = 28800
    num_iter_per_epoch = total_num_images / FLAGS.BATCH_SIZE

    # Start to train
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)
        modelfile = tf.train.latest_checkpoint(FLAGS.CHECKPOINT_SAVE_DIR)
        if modelfile:
            print('Restoring model from {}'.format(modelfile))
            epoch_cnt = int(re.split('-|_', modelfile)[-2]) + 1
            saver.restore(sess, modelfile)
        else:
            print('New model initilizing...')
            epoch_cnt = 1
            sess.run(tf.initialize_all_variables())

        sess.run(tf.initialize_local_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            while not coord.should_stop():
                _, loss_t, step = sess.run([train_op, total_loss, global_step])
                elapsed_time_per_iter = time.time() - start_time
                start_time = time.time()
                if step % 100 == 0:
                    print('Epoch {}, Iter {}, Loss = {}, Elpased_time: {}'.format(epoch_cnt, step, loss_t, elapsed_time_per_iter))
                if step % 1000 == 0:
                    # save current model
                    print('Save model at Epoch:{}, Iter:{}'.format(epoch_cnt, step))
                    saver.save(sess, os.path.join(FLAGS.CHECKPOINT_SAVE_DIR, 'fast-deblur-model_{}'.format(epoch_cnt)),
                        global_step=step)
                    # test & save current net output
                    out_ = sess.run(image_output)
                    with open(os.path.join(FLAGS.TEST_SAVE_DIR,'104055_{}.jpg'.format(step)), 'wb') as f:
                        f.write(out_)
                if step % num_iter_per_epoch == 0:
                    epoch_cnt += 1
        except tf.errors.OutOfRangeError:
            if step == None:
                print("QQQQ")
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
