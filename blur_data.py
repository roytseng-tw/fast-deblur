import numpy as np
import os
from os.path import join, basename
import glob
import struct
import imghdr
from scipy import ndimage, misc
import skimage
import time
import math
from joblib import Parallel, delayed
import multiprocessing


def get_info():
    max_sidelen = -1
    min_sidelen = 1000
    coco_train_path = '/home/roytseng/VisionNAS/coco_train2014'
    imagenames = glob.glob(join(coco_train_path,'*.jpg'))
    for idx, imagename in enumerate(imagenames):
        print(idx)
        h,w = get_image_size(imagename)
        tempmin, tempmax= sorted([h,w])
        if max_sidelen < tempmax:
            max_sidelen = tempmax
        if min_sidelen > tempmin:
            min_sidelen = tempmin
    print('Info: %d, %d' % (str(min_sidelen),str(max_sidelen)))

def get_info_fast():
    coco_train_path = '/home/roytseng/VisionNAS/coco_train2014'
    imagenames = glob.glob(join(coco_train_path,'*.jpg'))
    #imagenames = imagenames[:100]
    starttime = time.time()
    results = Parallel(n_jobs=10)(delayed(get_image_size)(x) for x in imagenames)
    elapsedtime = time.time() - starttime
    print(min(results), max(results))
    print(elapsedtime)

def get_image_size(fname):
    '''
    Determine the image type of fhandle and return its size.
    from draco
    '''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return height, width

def test_blur_image():
    blurfilter = np.loadtxt('Blur Filter.txt', delimiter='\t')
    blurfilter = np.expand_dims(blurfilter, axis=3)
    im = skimage.img_as_float(misc.imread('104055.jpg'))
    print(im[0,0,0])
    im_blurred = ndimage.filters.convolve(im, blurfilter, mode='constant', cval=0.0)
    misc.imsave('104055_blurred.jpg', im_blurred)

def blur_coco():
    blurfilter_2 = np.loadtxt('Blur Filter.txt', delimiter='\t')
    blurfilter_3 = np.expand_dims(blurfilter_2, axis=3)
    coco_train_path = '/home/roytseng/VisionNAS/coco_train2014'
    coco_blur_path = '/home/roytseng/VisionNAS/coco_blur2014'
    if not os.path.exists(coco_train_path):
        print('Source directory not found!')
        return
    if not os.path.exists(coco_blur_path):
        os.makedirs(coco_blur_path)
    impaths = glob.glob(join(coco_train_path,'*.jpg'))
    for idx, impath in enumerate(impaths):
        if idx < 4001:
            continue
        print(idx)
        filename = basename(impath)
        im = skimage.img_as_float(misc.imread(impath))
        if im.ndim == 2:
            im_blurred = ndimage.filters.convolve(im, blurfilter_2, mode='constant', cval=0.0)
        else:
            im_blurred = ndimage.filters.convolve(im, blurfilter_3, mode='constant', cval=0.0)
        misc.imsave(join(coco_blur_path,filename), im_blurred)
    print('finished')

if __name__ == '__main__':
    get_info_fast()