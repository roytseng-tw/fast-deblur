from os import listdir
from os.path import isfile, join
import os
import tensorflow as tf
import scipy.misc

mean_pixel = [123.68, 116.779, 103.939] # ImageNet average from VGG ..

def get_image(path):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    image = tf.sub(tf.to_float(image), mean_pixel)
    return image

def get_inputs_from_file(fin_path, batch_num, epochs, scale_factor=None):
    assert os.path.exists(fin_path) == True # check if input list file exits
    filename_queue = tf.train.string_input_producer( [fin_path], num_epochs=epochs)
    reader = tf.TextLineReader()
    _, line = reader.read(filename_queue)
    data_path, label_path = tf.decode_csv( line, record_defaults=[['data'], ['label']], field_delim=' ')
    # Assume input image format is JPEG
    data = tf.image.decode_jpeg( tf.read_file(data_path), channels=3)
    label = tf.image.decode_jpeg( tf.read_file(label_path), channels=3)
    # Crop to the same input size as VGG19(ImageNet)
    data_crop = tf.image.resize_image_with_crop_or_pad(data, 224, 224)
    label_crop = tf.image.resize_image_with_crop_or_pad(label, 224, 224)
    # Shrinks `data` to 1/4 size
    if scale_factor != None:
        new_size = (tf.to_int32(224*scale_factor), tf.to_int32(224*scale_factor))
        data_resize = tf.image.resize_images(data_crop, new_size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    else:
        data_resize = data_crop
    # Substract ImageNet mean, got from VGG19
    data_pre= tf.to_float(data_resize) - mean_pixel
    label_pre = tf.to_float(label_crop) - mean_pixel

    # Make batch
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_num
    data_batch, label_batch = tf.train.shuffle_batch(
        [data_pre, label_pre], batch_size=batch_num, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=4)
    return data_batch, label_batch
