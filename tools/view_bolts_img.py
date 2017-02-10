import os
import sys

import numpy as np
import skimage.io
import skimage.measure

from ..img_toolbox import io as tbx_io
from ..img_toolbox import bbox as tbx_bbox
from ..img_toolbox import plot as tbx_plt

PATCH_SIZE = [50, 50]
PATH = "/home/mw/Workspace/topo_data/"
SET = "set_02_falc"

if __name__ == '__main__':
    file_names = sorted(os.listdir(os.path.join(PATH, SET, "gt_class")))

    if len(sys.argv) > 1:
        file_names = [sys.argv[1]]

    imgs = tbx_io.load_images(os.path.join(PATH, SET, "images"), file_names)

    for i, f in enumerate(file_names):
        samples = []
        gt = skimage.io.imread(os.path.join(PATH, SET, "gt_class", f))
        labels, num = skimage.measure.label(gt, background=0, return_num=True, connectivity=2)

        for l in range(1, num):
            center = tbx_bbox.bbox_center(tbx_bbox.mask_to_bbox(labels == l))
            samples.append(np.array([i, center[0], center[1]]))

        samples = np.vstack(samples)
        print "name:", f, " #bolts:", samples.shape[0]
        tbx_plt.plot_patches(imgs, samples, PATCH_SIZE)
