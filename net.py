import os


def load_data():
    root = 'images/'
    imgs = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            imgs.append(os.path.join(path, name))
        
    return imgs

def extract_feats(img):
    print("Extract")




dataset = load_data()
print(dataset)    