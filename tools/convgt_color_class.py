import os
import sys

import skimage.io

PATH_SETS = "../../topo_data/"
SET = "set_02_falc"

PATH = os.path.join(PATH_SETS, SET)

if __name__ == '__main__':
    file_names = os.listdir(os.path.join(PATH, "gt_color"))

    if len(sys.argv) > 1:
        file_names = [sys.argv[1]]

    for f in file_names:
        label = skimage.io.imread(os.path.join(PATH, "gt_color", f))[:, :, 0]
        label[label < 10] = 1
        label[label == 255] = 0
        skimage.io.imsave(os.path.join(PATH, "gt_class", f), label)
