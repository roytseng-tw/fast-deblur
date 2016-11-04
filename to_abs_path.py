from os.path import join

if __name__ == '__main__':
    base_dir = '/home/roytseng/VisionNAS/EDOF-BSDS'
    list = []
    for line in open('/home/roytseng/VisionNAS/EDOF-BSDS/train_pair.lst', 'r'):
        list.append(line.split(' '))

    with open('/home/roytseng/VisionNAS/EDOF-BSDS/train_pair_tf.lst', 'w') as fout:
        for row in list:
            fout.write(join(base_dir,row[0]) + ' ' +join(base_dir,row[1]))