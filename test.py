
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
import skimage.filters

import util


marked = skimage.io.imread("../topo_data/example_set/marked/DJI_0126.JPG")[0]
label = util.marked_to_gt(marked)
image = skimage.io.imread("../topo_data/example_set/images/DJI_0126.JPG")[0]

coords = np.where(label)
print [marked[coords[0][i], coords[1][i]] for i in range(len(coords[0]))]


# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
# fig.set_size_inches(8, 3, forward=True)
# fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

# ax[0].imshow(marked)
# ax[1].imshow(label)
# # ax[0, 2].imshow(label)
# # ax[1, 0].imshow()
# # ax[1, 1].imshow()
# # ax[1, 2].imshow()

# plt.show()
