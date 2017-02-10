import os

import numpy as np
import skimage.io

PATH_SETS = "../../topo_data/"
SET = "set_02_falc"

PATH = os.path.join(PATH_SETS, SET)
file_names = os.listdir(os.path.join(PATH, "gt_class"))

for f in file_names:
    label = skimage.io.imread(os.path.join(PATH, "gt_class", f))
    label = np.tile(np.expand_dims(label, axis=2), 3)
    label[label == 0] = 255
    label[label == 1] = 0
    skimage.io.imsave(os.path.join(PATH, "gt_color", f), label)
