from os import listdir
from os.path import isfile, join
import os
import tensorflow as tf
import scipy.misc

mean_pixel = [123.68, 116.779, 103.939] # ImageNet average from VGG ..

'''
resize and subtract-mean 
'''
def preprocess(image, size, is_max_length):
    # resize image
    if size:
        shape = tf.shape(image)
        size_t = tf.constant(size, tf.float32)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)

        # if True:  use size as max size
        # if False: use size as min size 
        pred = tf.less(width, height) if is_max_length else tf.less(height, width)

        new_height, new_width = tf.cond(
            pred,
            lambda: (size_t, (width * size_t) / height),
            lambda: ((height * size_t) / width, size_t))
        image = tf.image.resize_images(image, tf.to_int32(new_height), tf.to_int32(new_width))

    # subtract mean_pixel
    if mean_pixel:
        normalised_image = tf.to_float(image) - mean_pixel
    return normalised_image

def preprocess_2(image, scale):
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    new_height, new_width = (height * scale, width * scale)
    image = tf.image.resize_images(image, tf.to_int32(new_height), tf.to_int32(new_width))
    normalised_image = image - mean_pixel
    return normalised_image

'''
Get an image as network input
max_length: Wether size dictates longest or shortest side. Default longest
size: `None` denotes no resize
'''
def get_image(path, size=None, is_max_length=True):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    image = tf.sub(tf.to_float(image), mean_pixel)
    return image

'''
Get images under `image_dir`
'''
def get_images(batch_num, image_dir, epochs, size=None, shuffle=True, crop=False):
    filenames = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    if not shuffle:
        filenames = sorted(filenames)

    isPng = filenames[0].lower().endswith('png') # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs, shuffle=shuffle)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if isPng else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = preprocess(image, size, False)
    if not crop:
        return tf.train.batch([processed_image], batch_num, dynamic_pad=True)

    cropped_image = tf.slice(processed_image, [0,0,0], [size, size, 3])
    cropped_image.set_shape((size, size, 3))
    images = tf.train.batch([cropped_image], batch_num)
    return images

def get_inputs(input_dir, input_file, batch_num, epochs, size):
    assert os.path.exists(join(input_dir,input_file)) == True # check if input list file exits
    filename_queue = tf.train.string_input_producer([join(input_dir,input_file)], num_epochs=epochs)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    datapath, labelpath = tf.decode_csv(value, record_defaults=[['data'], ['label']], field_delim=' ')
    data = tf.image.decode_jpeg(tf.read_file(datapath), channels=3)   # assume is JPEG
    label = tf.image.decode_jpeg(tf.read_file(labelpath), channels=3) # assume is JPEG
    data = tf.image.resize_image_with_crop_or_pad(data, 224, 224)    # intentionally set to the same input size as vgg19
    label = tf.image.resize_image_with_crop_or_pad(label, 224, 224)
    # data = tf.image.resize_images(data, 64, 64, method=tf.image.ResizeMethod.BILINEAR, align_corners=False) # shrinks to 1/4
    # Substract vgg ILSVRC mean
    data = tf.to_float(data) - mean_pixel
    label = tf.to_float(label) - mean_pixel
    
    # prepare batch
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_num
    data_batch, label_batch = tf.train.shuffle_batch(
        [data, label], batch_size=batch_num, capacity=capacity, 
        min_after_dequeue=min_after_dequeue, num_threads=4)
    return data_batch, label_batch
    
    '''
    output_data = tf.image.encode_jpeg(tf.squeeze(tf.saturate_cast( tf.add(data_batch, mean_pixel), tf.uint8)))
    output_label = tf.image.encode_jpeg(tf.squeeze(tf.saturate_cast( tf.add(label_batch, mean_pixel), tf.uint8)))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            out1, out2, path1, path2 = sess.run([output_data, output_label, datapath, labelpath])
            print([path1,path2])
            with open('test_data.jpg', 'wb') as f:
                f.write(out1)
            with open('test_label.jpg', 'wb') as f:
                f.write(out2)
        coord.request_stop()
        coord.join(threads)
    '''

if __name__ == '__main__':
    get_inputs('/home/roytseng/VisionNAS/EDOF-BSDS','train_pair_tf.lst', 1, 1, None)