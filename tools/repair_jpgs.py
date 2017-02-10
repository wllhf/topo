import os

import skimage.io

PATH_SETS = "../topo_data/"
SET = "set_02_falc"

PATH = os.path.join(PATH_SETS, SET, "images")
file_names = os.listdir(PATH)

for f in file_names:
    im = skimage.io.imread(os.path.join(PATH, f))[0]
    skimage.io.imsave(os.path.join(PATH, f), im)
