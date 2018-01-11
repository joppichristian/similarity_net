from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import functions_incresnet as f
import time

features = np.load('feats_incresnet/features_ie.npy')
imgs = np.load('feats_incresnet/im_list_ie.npy')
dists = np.load('feats_incresnet/dist_ie.npy')
net_name = "Inception ResNet V2"


def test(features,imgs,dists):
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

    imgs_good = list(set(imgs)-set(no_img))
    indexes = []
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Pantaloni' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Gonne' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Giacche' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Camicie' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Maglie' in im ])
    indexes.append([i for im,i in zip(imgs_good,range(0,len(imgs_good))) if 'Vestiti' in im ])

    maps = []
    maps_tot = []
    for ind in indexes:
    	means_acc_pr = []
    	for g,i in zip(good_pos,range(0,len(good_pos))):
		if i in ind:
			tmp = 0.0
			for gg,idx in zip(g,range(1,len(g)+1)):
				tmp = tmp + float(idx)/(gg+1)
			m_a_p = tmp/len(g)
			maps_tot.append(m_a_p)
			means_acc_pr.append(m_a_p)
	mean_acc_precision=np.mean(means_acc_pr)
        maps.append(mean_acc_precision)
    print('Pantaloni:' + str(maps[0]))
    print('Gonne:' + str(maps[1]))
    print('Giacche:' + str(maps[2]))
    print('Camicie:' + str(maps[3]))
    print('Maglie:' + str(maps[4]))
    print('Vestiti:' + str(maps[5]))
    print("\nGeneral:" + str(np.mean(maps_tot)))
	



test(features,imgs,dists)


