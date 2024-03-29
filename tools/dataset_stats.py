import os

import skimage.io
import skimage.measure

PATH = "../topo_data/"
SET = "set_02_falc"

gt_file_names = sorted(os.listdir(os.path.join(PATH, SET, "gt_class")))

nbolts = []

for f in gt_file_names:
    gt = skimage.io.imread(os.path.join(PATH, SET, "gt_class", f))
    labels, num = skimage.measure.label(gt, background=0, return_num=True, connectivity=2)
    nbolts.append(num)


print "name   : ", SET
print "#images: ", len(gt_file_names)
print "#bolts : ", sum(nbolts)
