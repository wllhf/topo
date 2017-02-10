import os

import numpy as np
import skimage.io
import skimage.measure

from ..img_toolbox import io as tbx_io
from ..img_toolbox import bbox as tbx_bbox
from ..img_toolbox import sample as tbx_sample

PATH_02 = "topo_data/set_02_falc"
PATH_03 = "topo_data/set_03"

PATCH_SIZE = [50, 50]
RATIO = 2.0

if __name__ == '__main__':
    file_names = sorted(os.listdir(os.path.join(PATH_02, "gt_class")))
    imgs = tbx_io.load_images(os.path.join(PATH_02, "images"), file_names)

    # sample
    samples_t = []
    samples_f = []
    for i, f in enumerate(file_names):
        gt = skimage.io.imread(os.path.join(PATH_02, "gt_class", f))

        # true
        labels, num = skimage.measure.label(gt, background=0, return_num=True, connectivity=2)
        for l in range(1, num):
            center = tbx_bbox.bbox_center(tbx_bbox.mask_to_bbox(labels == l))
            samples_t.append(np.array([i, center[0], center[1]]))

        # false
        coords = tbx_sample.grid_sample_coords(gt.shape, [10, 10], PATCH_SIZE)
        coords = coords[tbx_sample.at(gt, coords, PATCH_SIZE, flatten=True).sum(axis=1) == 0]
        samples_f.append(np.hstack([np.ones((coords.shape[0], 1), coords.dtype) * i, coords]))

    samples_t, samples_f = np.vstack(samples_t), np.vstack(samples_f)

    # generate bolt samples
    patches_t = []
    for i, img in enumerate(imgs):
        patches_t.append(tbx_sample.at(img, samples_t[samples_t[:, 0] == i, 1:], PATCH_SIZE, True, True, False))
    patches_t = np.vstack(patches_t)

    # subsample
    indices = np.random.randint(samples_f.shape[0], size=int(patches_t.shape[0] * RATIO))
    samples_f = samples_f[indices, :]

    # generate wall samples
    patches_f = []
    for i, img in enumerate(imgs):
        patches_f.append(tbx_sample.at(img, samples_f[samples_f[:, 0] == i, 1:], PATCH_SIZE, True, True, False))
    patches_f = np.vstack(patches_f)

    # stack em up
    labels = np.vstack([np.ones((patches_t.shape[0], 1), dtype='uint8'), np.zeros((patches_f.shape[0], 1), dtype='uint8')])
    patches = np.vstack([patches_t, patches_f])

    # save set
    if not os.path.exists(PATH_03):
        os.makedirs(PATH_03)

    np.save(os.path.join(PATH_03, "labels"), labels)
    np.save(os.path.join(PATH_03, "patches"), patches)
