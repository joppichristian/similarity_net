from utils import functions as f
import numpy as np
from scipy.spatial.distance import cdist







imgs,json_dataset = f.load_data('../DatasetNL/','../DatasetNL/results/')



feats = f.get_features(imgs,json_dataset)
features = np.squeeze(np.asarray(feats))
dists = cdist(features,features,'cosine')

np.save('dist.npy',dists)
np.save('features.npy',features)
np.save('im_list.npy',imgs)