from utils import extract_feat as ef
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""imgs,feats = ef.get_features()
features = np.squeeze(np.asarray(feats))
np.save('features.npy',features)
np.save('im_list.npy',imgs)"""

features = np.load('features.npy')
imgs = np.load('im_list.npy')
dists = cdist(features,features,'cosine')

im_ind = 11
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