from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2
from nets import vgg

from preprocessing import vgg_preprocessing

def load_data():
    root = 'images/'
    imgs = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            imgs.append(os.path.join(path, name))
        
    return imgs






def get_features():
    dataset = load_data()


    url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

    # Specify where you want to download the model to
    checkpoints_dir = 'model'

    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

    slim = tf.contrib.slim

    im_size = vgg.vgg_16.default_image_size

    features = []

    for img in dataset:
        with tf.Graph().as_default():
            im = tf.image.decode_image(tf.read_file(img),channels=3)
            pr_im = vgg_preprocessing.preprocess_image(im,im_size, im_size,is_training=False)
            pr_im  = tf.expand_dims(pr_im, 0)
            
            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, _ = vgg.vgg_16(pr_im,
                                num_classes=0,
                                is_training=False)
            
            init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
            slim.get_model_variables('vgg_16'))

            with tf.Session() as sess:
                init_fn(sess)
                im = sess.run(im)
                pr_im = sess.run(pr_im)
                logits = sess.run(logits)
            features.append(logits)
        tf.reset_default_graph()
    print(len(features))

    return dataset,features 