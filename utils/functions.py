from matplotlib import pyplot as plt

import numpy as np
import os, json
import tensorflow as tf
import urllib3
from nets import vgg
from shutil import copyfile
from preprocessing import vgg_preprocessing
import matplotlib.patches as patches
from sklearn.metrics import auc
import time
GPU_MEMORY = 1

gpu_options = tf.GPUOptions( 
    per_process_gpu_memory_fraction = GPU_MEMORY, 
    allow_growth = True ) 
    
config = tf.ConfigProto( 
    device_count = { 'GPU': 1 }, 
    gpu_options = gpu_options )  

def load_data(root_imgs,root_bbs):
    root = 'images/'
    imgs = []
    json_data = []
    to_download = []
    for path, subdirs, files in os.walk(root_bbs):
        for name in files:
            if 'NO_' not in name and '.json' in name:
                dir_path = os.path.join(path.split('/')[3],path.split('/')[4],path.split('/')[5])
                json_path = os.path.join(path,name)
                name_img = name.replace('.json','.jpg')
                im_path = os.path.join(root_imgs,dir_path,name_img)
                if not os.path.exists(root+dir_path):
                    os.makedirs(root+dir_path)
                copyfile(im_path,root+dir_path+'/'+name_img)
                copyfile(json_path,root+dir_path+'/'+name)
        
    for path, subdirs, files in os.walk(root):
        for name in files:
            if '.jpg' in name:
                imgs.append(os.path.join(path, name))
                js = json.load(open(os.path.join(path, name.replace('.jpg','.json'))))
                json_data.append(js) 
    return imgs, json_data




def test_crop(dataset,bbs):
    i=1
    t1 = time.time()
    for img,bb in zip(dataset,bbs):
        print(i,img)
        bbs_inside = bb['bb_inside']
        bbs_outside = bb['bb']
        with tf.Graph().as_default():
            im_decode = tf.image.decode_image(tf.read_file(img),channels=3)
            try:
                im = tf.image.crop_to_bounding_box(im_decode,bbs_inside['min_y'],bbs_inside['min_x'],bbs_inside['max_y']-bbs_inside['min_y'],bbs_inside['max_x']-bbs_inside['min_x'])
                with tf.Session(config=config) as sess:
                    im_decode = sess.run(im_decode)
                    im = sess.run(im)
            except:
                print("Errore di croping")
                im = tf.image.crop_to_bounding_box(im_decode,bbs_outside['min_y'],bbs_outside['min_x'],bbs_outside['max_y']-bbs_outside['min_y'],bbs_outside['max_x']-bbs_outside['min_x'])
                with tf.Session(config=config) as sess:
                    im_decode = sess.run(im_decode)
                    im = sess.run(im)
        i=i+1
    t = time.time() - t1
    print(t1)

def get_features(dataset,bbs,type_features):

    url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
    checkpoints_dir = 'model'
    slim = tf.contrib.slim
    im_size = vgg.vgg_16.default_image_size
    features = []
    if type_features=='e':
        i = 1
        for img,bb in zip(dataset,bbs):
            
            print(i)
            bb = bb['bb']
            with tf.Graph().as_default():
                im_decode = tf.image.decode_image(tf.read_file(img),channels=3)
                im = tf.image.crop_to_bounding_box(im_decode,bb['min_y'],bb['min_x'],bb['max_y']-bb['min_y'],bb['max_x']-bb['min_x'])
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
                    im_decode = sess.run(im_decode)
                    im = sess.run(im)
                    pr_im = sess.run(pr_im)
                    logits = sess.run(logits)
                logits = np.squeeze(np.asarray(logits))
                features.append(logits)
            tf.reset_default_graph()
            i = i+1
            np.save('tmp',features)
            np.save('tmp_i',i)
    elif type_features=='i':
        i = 1
        for img,bb in zip(dataset,bbs):
            
            print(i)
            bb_inside = bb['bb_inside']
            bb = bb['bb']
            with tf.Graph().as_default():
                try:
                    im_decode = tf.image.decode_image(tf.read_file(img),channels=3)
                    im = tf.image.crop_to_bounding_box(im_decode,bb_inside['min_y'],bb_inside['min_x'],bb_inside['max_y']-bb_inside['min_y'],bb_inside['max_x']-bb_inside['min_x'])
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
                        im_decode = sess.run(im_decode)
                        im = sess.run(im)
                        pr_im = sess.run(pr_im)
                        logits = sess.run(logits)
                    logits = np.squeeze(np.asarray(logits))
                    features.append(logits)
                except:
                    
                    im_decode = tf.image.decode_image(tf.read_file(img),channels=3)
                    im = tf.image.crop_to_bounding_box(im_decode,bb['min_y'],bb['min_x'],bb['max_y']-bb['min_y'],bb['max_x']-bb['min_x'])
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
                        im_decode = sess.run(im_decode)
                        im = sess.run(im)
                        pr_im = sess.run(pr_im)
                        logits = sess.run(logits)
                    logits = np.squeeze(np.asarray(logits))
                    features.append(logits)
            tf.reset_default_graph()
            i = i+1
            np.save('tmp',features)
            np.save('tmp_i',i)
    elif type_features == 'ie':
        
        i = 1
        for img,bb in zip(dataset,bbs):
            feat = []
            print(i)
            bb_inside = bb['bb_inside']
            bb = bb['bb']
            with tf.Graph().as_default():
                im_decode = tf.image.decode_image(tf.read_file(img),channels=3)
                im = tf.image.crop_to_bounding_box(im_decode,bb['min_y'],bb['min_x'],bb['max_y']-bb['min_y'],bb['max_x']-bb['min_x'])
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
                    im_decode = sess.run(im_decode)
                    im = sess.run(im)
                    pr_im = sess.run(pr_im)
                    logits = sess.run(logits)
                
                logits = np.squeeze(np.asarray(logits))
                feat = list(logits)
            tf.reset_default_graph()
            with tf.Graph().as_default():
                try:
                    im_decode = tf.image.decode_image(tf.read_file(img),channels=3)
                    im = tf.image.crop_to_bounding_box(im_decode,bb_inside['min_y'],bb_inside['min_x'],bb_inside['max_y']-bb_inside['min_y'],bb_inside['max_x']-bb_inside['min_x'])
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
                        im_decode = sess.run(im_decode)
                        im = sess.run(im)
                        pr_im = sess.run(pr_im)
                        logits = sess.run(logits)
                    logits = np.squeeze(np.asarray(logits))
                    feat = feat + list(logits)
                except:
                    print("errore")
                    im_decode = tf.image.decode_image(tf.read_file(img),channels=3)
                    im = tf.image.crop_to_bounding_box(im_decode,bb['min_y'],bb['min_x'],bb['max_y']-bb['min_y'],bb['max_x']-bb['min_x'])
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
                        im_decode = sess.run(im_decode)
                        im = sess.run(im)
                        pr_im = sess.run(pr_im)
                        logits = sess.run(logits)
                    logits = np.squeeze(np.asarray(logits))
                    feat = feat + list(logits)
            tf.reset_default_graph()
            features.append(feat)
            i = i+1
            np.save('tmp',features)
            np.save('tmp_i',i)
    return features 



def cmc(k_n,K):
    accuracy = []
    for k in range(1,K):
        accuracy.append(float(len([i for i in k_n if i <= k]))/len(k_n))

    _auc = auc(range(1,K),accuracy)
    
    return accuracy,_auc