import os

import numpy as np
import skimage.io

FOLDER_MARKED = "../topo_data/example_set/marked"
FOLDER_LABELS = "../topo_data/example_set/ground_truth"
LABEL_COLOR = (220, 100, 100)
LOWER_BOUND = (True, False, False)


def marked_to_gt(img, clr=(254, 0, 0)):
    matching = [img[:, :, 0] == clr[0], img[:, :, 1] == clr[1], img[:, :, 2] == clr[2]]
    return np.logical_and.reduce(matching).astype('uint8')*254


def marked_to_gt_th(img, clr=(220, 0, 0), bounds=(True, False, False)):
    r = img[:, :, 0] >= clr[0] if bounds[0] else img[:, :, 0] <= clr[0]
    g = img[:, :, 1] >= clr[1] if bounds[1] else img[:, :, 1] <= clr[1]
    b = img[:, :, 2] >= clr[2] if bounds[2] else img[:, :, 2] <= clr[2]
    return np.logical_and.reduce([r, g, b]).astype('uint8')*254


if __name__ == '__main__':

    if not os.path.exists(FOLDER_LABELS):
        os.makedirs(FOLDER_LABELS)

    file_names = os.listdir(FOLDER_MARKED)

    for name in file_names:
        img = skimage.io.imread(os.path.join(FOLDER_MARKED, name))
        img = img[0] if len(img.shape) < 3 else img  # not sure why this is necessary
        label = marked_to_gt_th(img, LABEL_COLOR, LOWER_BOUND)
        skimage.io.imsave(os.path.join(FOLDER_LABELS, name), label)
