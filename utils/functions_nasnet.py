from matplotlib import pyplot as plt

import numpy as np
import os, json
import tensorflow as tf
import urllib3
from nets import vgg
from nets import nasnet
from shutil import copyfile
from preprocessing import inception_preprocessing
import matplotlib.patches as patches
from sklearn.metrics import auc
import time


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



def get_features(dataset,bbs,type_features):

    url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
    checkpoints_dir = 'model'
    slim = tf.contrib.slim
    im_size = nasnet.build_nasnet_large.default_image_size
    features = []
    if type_features=='e':
            network = tf.Graph()
            with network.as_default():
                img = tf.placeholder(tf.string)
                bb_xmin = tf.placeholder(tf.int32)
                bb_ymin = tf.placeholder(tf.int32)
                bb_xmax = tf.placeholder(tf.int32)
                bb_ymax = tf.placeholder(tf.int32)
                im_decode = tf.image.decode_jpeg(tf.read_file(img),channels=3)
                im = tf.image.crop_to_bounding_box(im_decode,bb_ymin,bb_xmin,bb_ymax-bb_ymin,bb_xmax-bb_xmin)
                pr_im = inception_preprocessing.preprocess_image(im,im_size, im_size,is_training=False)
                pr_im  = tf.expand_dims(pr_im, 0)
                
                with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
                    logits, _ = nasnet.build_nasnet_large(pr_im,
                                    num_classes=0,
                                    is_training=False)
                
                init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoints_dir, 'nasnet.ckpt'),
                slim.get_model_variables())

            with tf.Session(graph=network) as sess:
                i = 1
                sess.run(tf.global_variables_initializer())
                init_fn(sess)
                for img_p,bb in zip(dataset,bbs):   
                    print(i)
                    b = bb['bb']
                    _,_,_,l = sess.run([im_decode,im,pr_im,logits],feed_dict={img:img_p,bb_xmin:b['min_x'],bb_ymin:b['min_y'],bb_xmax:b['max_x'],bb_ymax:b['max_y']})   
                    l = np.squeeze(np.asarray(l))
                    features.append(l)
                    i = i+1
                    np.save('tmp',features)
                    np.save('tmp_i',i)
            return features
    elif type_features=='i':
            network = tf.Graph()
            with network.as_default():
                img = tf.placeholder(tf.string)
                bb_xmin = tf.placeholder(tf.int32)
                bb_ymin = tf.placeholder(tf.int32)
                bb_xmax = tf.placeholder(tf.int32)
                bb_ymax = tf.placeholder(tf.int32)
                im_decode = tf.image.decode_jpeg(tf.read_file(img),channels=3)
                im = tf.image.crop_to_bounding_box(im_decode,bb_ymin,bb_xmin,bb_ymax-bb_ymin,bb_xmax-bb_xmin)
                pr_im = inception_preprocessing.preprocess_image(im,im_size, im_size,is_training=False)
                pr_im  = tf.expand_dims(pr_im, 0)
                
                with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
                    logits, _ = nasnet.build_nasnet_large(pr_im,
                                    num_classes=0,
                                    is_training=False)
                
                init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoints_dir, 'nasnet.ckpt'),
                slim.get_model_variables())
                
            with tf.Session(graph=network) as sess:
                i = 1
                sess.run(tf.global_variables_initializer())
                init_fn(sess)
                for img_p,bb in zip(dataset,bbs):   
                    print(i)
                    b_i = bb['bb_inside']
                    b = bb['bb']
                    try:
                        _,_,_,l = sess.run([im_decode,im,pr_im,logits],feed_dict={img:img_p,bb_xmin:b_i['min_x'],bb_ymin:b_i['min_y'],bb_xmax:b_i['max_x'],bb_ymax:b_i['max_y']})   
                        l = np.squeeze(np.asarray(l))
                        features.append(l)
                    except:
                        _,_,_,l = sess.run([im_decode,im,pr_im,logits],feed_dict={img:img_p,bb_xmin:b['min_x'],bb_ymin:b['min_y'],bb_xmax:b['max_x'],bb_ymax:b['max_y']})   
                        l = np.squeeze(np.asarray(l))
                        features.append(l)
                    i = i+1
                    np.save('tmp',features)
                    np.save('tmp_i',i)
            return features
    elif type_features == 'ie':
            network = tf.Graph()
            with network.as_default():
                img = tf.placeholder(tf.string)
                bb_xmin = tf.placeholder(tf.int32)
                bb_ymin = tf.placeholder(tf.int32)
                bb_xmax = tf.placeholder(tf.int32)
                bb_ymax = tf.placeholder(tf.int32)
                im_decode = tf.image.decode_jpeg(tf.read_file(img),channels=3)
                im = tf.image.crop_to_bounding_box(im_decode,bb_ymin,bb_xmin,bb_ymax-bb_ymin,bb_xmax-bb_xmin)
                pr_im = inception_preprocessing.preprocess_image(im,im_size, im_size,is_training=False)
                pr_im  = tf.expand_dims(pr_im, 0)
                
                with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
                    logits, _ = nasnet.build_nasnet_large(pr_im,
                                    num_classes=0,
                                    is_training=False)
                
                init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoints_dir, 'nasnet.ckpt'),
                slim.get_model_variables())
                


            with tf.Session(graph=network) as sess:
                i = 1       
                sess.run(tf.global_variables_initializer())
                init_fn(sess)
                for img_p,bb in zip(dataset,bbs):   
                    print(i)
                    b_i = bb['bb_inside']
                    b = bb['bb']
                    _,_,_,l = sess.run([im_decode,im,pr_im,logits],feed_dict={img:img_p,bb_xmin:b['min_x'],bb_ymin:b['min_y'],bb_xmax:b['max_x'],bb_ymax:b['max_y']})   
                    l = np.squeeze(np.asarray(l))
                    feat = list(l)
                    try:
                        _,_,_,l = sess.run([im_decode,im,pr_im,logits],feed_dict={img:img_p,bb_xmin:b_i['min_x'],bb_ymin:b_i['min_y'],bb_xmax:b_i['max_x'],bb_ymax:b_i['max_y']})   
                        l = np.squeeze(np.asarray(l))
                        feat = feat+ list(l)
                    except:
                        _,_,_,l = sess.run([im_decode,im,pr_im,logits],feed_dict={img:img_p,bb_xmin:b['min_x'],bb_ymin:b['min_y'],bb_xmax:b['max_x'],bb_ymax:b['max_y']})   
                        l = np.squeeze(np.asarray(l))
                        feat = feat + list(l)
                    i = i+1
                    features.append(feat)
                    np.save('tmp',features)
                    np.save('tmp_i',i)
            return features 



def cmc(k_n,K):
    accuracy = []
    for k in range(1,K):
        accuracy.append(float(len([i for i in k_n if i <= k]))/len(k_n))

    _auc = auc(range(1,K),accuracy)
    
    return accuracy,_auc