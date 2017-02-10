from __future__ import division

import os
import random

FOLDER_SETS = "../topo_data/"
SET = "set_02_falc"

file_names = os.listdir(os.path.join(FOLDER_SETS, SET, "gt_class"))

size_train = len(file_names) // 2
random.shuffle(file_names)
train = file_names[:size_train]
test = file_names[size_train:]

with open(os.path.join(FOLDER_SETS, SET, "splits", 'train.txt'), 'w+') as f:
    for item in train:
        print>>f, item

with open(os.path.join(FOLDER_SETS, SET, "splits", 'test.txt'), 'w+') as f:
    for item in test:
        print>>f, item
