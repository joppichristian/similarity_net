import numpy as np


features = np.load('feats_incresnet/features_e.npy')
imgs = np.load('feats_incresnet/im_list_e.npy')

pantaloni = []
gonne = []
vestiti = []
maglie = []
camicie = []
giacche = []


i_pantaloni = []
i_gonne = []
i_vestiti = []
i_maglie = []
i_camicie = []
i_giacche = []

for f,i in zip(features,imgs):
    if 'Pantaloni' in i:
        pantaloni.append(f)
        i_pantaloni.append(i)
    elif 'Gonne' in i:
        gonne.append(f)
        i_gonne.append(i)
    elif 'Vestiti' in i:
        vestiti.append(f)
        i_vestiti.append(i)
    elif 'Maglie'in i:
        maglie.append(f)
        i_maglie.append(i)
    elif 'Camicie' in i:
        camicie.append(f)
        i_camicie.append(i)
    else:
        giacche.append(f)
        i_giacche.append(i)

print(len(pantaloni))
print(len(gonne))
print(len(vestiti))
print(len(maglie))
print(len(camicie))
print(len(giacche))

np.save('features_byCat/pantaloni.npy',pantaloni)
np.save('features_byCat/gonne.npy',gonne)
np.save('features_byCat/vestiti.npy',vestiti)
np.save('features_byCat/maglie.npy',maglie)
np.save('features_byCat/camicie.npy',camicie)
np.save('features_byCat/giacche.npy',giacche)


np.save('features_byCat/imgs_pantaloni.npy',i_pantaloni)
np.save('features_byCat/imgs_gonne.npy',i_gonne)
np.save('features_byCat/imgs_vestiti.npy',i_vestiti)
np.save('features_byCat/imgs_maglie.npy',i_maglie)
np.save('features_byCat/imgs_camicie.npy',i_camicie)
np.save('features_byCat/imgs_giacche.npy',i_giacche)
