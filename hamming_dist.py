import numpy as np
from scipy.spatial.distance import cdist
import time

features = np.load('feats_incresnet/features_e.npy')


for f in features:
    f[f!=0] = 1

t1 = time.time()

dists = cdist(features,features,'hamming')

t = time.time() - t1

print(t)

np.save('feats_incresnet/dist_ham_e.npy',dists)