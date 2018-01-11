from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import functions_incresnet as f
import time

features = np.load('features_ie.npy')
imgs = np.load('im_list_ie.npy')
dists = np.load('dist_ie.npy')
net_name = "Inception ResNet V2"


def all_class_test(features,imgs,dists):
    good_pos = []
    no_img = []
    for i,im in zip(range(0,len(imgs)),imgs):
        ranking_index = np.argsort(dists[i,:])
        similars = []
        for j in range(1,len(ranking_index)):
            similars.append(imgs[ranking_index[j]])
        dress = im.split('/').pop().split('_')[0]
        tmp = []
        for j,s in zip(range(0,len(similars)),similars):
            dr_s = s.split('/').pop().split('_')[0]
            if dress == dr_s: 
                tmp.append(j)
        if len(tmp) == 0:
            no_img.append(im)
        else:
            good_pos.append(tmp)

    first_pos = [g[0] for g in good_pos]

    imgs_good = list(set(imgs)-set(no_img))




    K = len(first_pos)


    acc, auc = f.cmc(first_pos,K)
    auc = round(auc/len(first_pos),4)


    fig = plt.figure()
    fig.suptitle('CMC' , fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title(net_name + ' -- K='+str(K))

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')


    ax.plot(range(0,len(acc)),acc)
    

    #FOR THE UNIQUE CURVE
    ax.text(len(first_pos)/2, 0.7, u'AUC : ' + str(auc),
       bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})

    plt.draw()
    plt.pause(1)
    raw_input("<Hit Enter To Close>")
    plt.close()



def separate_class_test(features,imgs,dists):
    good_pos = []
    no_img = []
    for i,im in zip(range(0,len(imgs)),imgs):
        ranking_index = np.argsort(dists[i,:])
        similars = []
        for j in range(1,len(ranking_index)):
            similars.append(imgs[ranking_index[j]])
        dress = im.split('/').pop().split('_')[0]
        tmp = []
        for j,s in zip(range(0,len(similars)),similars):
            dr_s = s.split('/').pop().split('_')[0]
            if dress == dr_s: 
                tmp.append(j)
        if len(tmp) == 0:
            no_img.append(im)
        else:
            good_pos.append(tmp)

    first_pos = [g[0] for g in good_pos]
    imgs_good = list(set(imgs)-set(no_img))
    indexes = []
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Pantaloni' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Gonne' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Giacche' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Camicie' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Maglie' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Vestiti' in im ])



    K = len(first_pos)

    #ONE CMC FOR ONE CATEGORY
    accs = []
    aucs =  []
    for ind in indexes:
        tmp_pos = [first_pos[i] for i in ind]
        acc, auc = f.cmc(tmp_pos,K)
        auc = round(auc/len(first_pos),4)
        accs.append(acc)
        aucs.append(auc)


    fig = plt.figure()
    fig.suptitle('CMC' , fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title(net_name + ' -- K='+str(K))

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')

    #ONE CMC FOR ONE CATEGORY
    plots = []
    for acc,auc in zip(accs,aucs):
        pl, = ax.plot(range(0,len(acc)),acc)
        plots.append(pl)

    #FOR THE UNIQUE CURVE
        #ax.text(len(first_pos)/2, 0.7, u'AUC : ' + str(),
        #    bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
    plt.legend(plots,['Pantaloni - AUC: '+ str(aucs[0]) ,'Gonne - AUC: '+ str(aucs[1]) ,'Giacche - AUC: '+ str(aucs[2]),
        'Camicie - AUC: '+ str(aucs[3]),'Maglie - AUC: '+ str(aucs[4]),'Vestiti - AUC: '+ str(aucs[5])])


    plt.draw()
    plt.pause(1)
    raw_input("<Hit Enter To Close>")
    plt.close()



separate_class_test(features,imgs,dists)
all_class_test(features,imgs,dists)
#print(features.shape)
"""


im_ind = 17
print(good_pos[im_ind])

ranking_index = np.argsort(dists[im_ind,:])
print(imgs[im_ind])
print(imgs[ranking_index[1:5]])



fig = plt.figure()
a=fig.add_subplot(1,6,1)
plt.imshow(mpimg.imread(imgs[im_ind]))
a=fig.add_subplot(1,6,2)
plt.imshow(mpimg.imread(imgs[ranking_index[1]]))
a=fig.add_subplot(1,6,3)
plt.imshow(mpimg.imread(imgs[ranking_index[2]]))
a=fig.add_subplot(1,6,4)
plt.imshow(mpimg.imread(imgs[ranking_index[3]]))
a=fig.add_subplot(1,6,5)
plt.imshow(mpimg.imread(imgs[ranking_index[4]]))
a=fig.add_subplot(1,6,6)
plt.imshow(mpimg.imread(imgs[ranking_index[5]]))
plt.show()

"""
