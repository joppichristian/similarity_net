from utils import functions_incV4 as f
import numpy as np
from scipy.spatial.distance import cdist




imgs,json_dataset = f.load_data('../DatasetNL','../DatasetNL/results')
"""
feats = f.get_features(imgs,json_dataset,'ie')
dists = cdist(feats,feats,'cosine')

np.save('dist_ie.npy',dists)
np.save('features_ie.npy',feats)
np.save('im_list_ie.npy',imgs)
"""

feats = f.get_features(imgs,json_dataset,'i')
dists = cdist(feats,feats,'cosine')

np.save('dist_i.npy',dists)
np.save('features_i.npy',feats)
np.save('im_list_i.npy',imgs)

feats = f.get_features(imgs,json_dataset,'e')
dists = cdist(feats,feats,'cosine')

np.save('dist_e.npy',dists)
np.save('features_e.npy',feats)
np.save('im_list_e.npy',imgs)

