import numpy as np


def marked_to_gt(img, th=220):
    gt = np.zeros(img.shape)
    print (img[:, :, 1] == 0).sum()
    gt[img[:, :, 0] > th] = 1
    return gt


def get_bolt_coords():
    pass
