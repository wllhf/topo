import os

import numpy as np
import skimage.io
import skimage.measure

from ..img_toolbox import io as tbx_io
from ..img_toolbox import bbox as tbx_bbox
from ..img_toolbox import plot as tbx_plt

PATCH_SIZE = [50, 50]
PATH = "/home/mw/Workspace/topo_data/"
SET = "set_02_falc"

file_names = sorted(os.listdir(os.path.join(PATH, SET, "gt_class")))

samples = []

for i, f in enumerate(file_names):
    gt = skimage.io.imread(os.path.join(PATH, SET, "gt_class", f))
    labels, num = skimage.measure.label(gt, background=0, return_num=True, connectivity=2)

    for l in range(num):
        center = tbx_bbox.bbox_center(tbx_bbox.mask_to_bbox(labels == l))
        samples.append(np.array([i, center[0], center[1]]))

samples = np.vstack(samples)
print samples.shape

imgs = tbx_io.load_images(os.path.join(PATH, SET, "images"), file_names)
tbx_plt.plot_patches(imgs, samples[-100:, :], PATCH_SIZE)
