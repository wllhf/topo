from __future__ import division
from __future__ import print_function

import os
import random

FOLDER_SETS = "topo_data/"
SET = "set_02_falc"

RATIO_TRAIN = 0.6
RATIO_VALID = 0.1

file_names = os.listdir(os.path.join(FOLDER_SETS, SET, "gt_class"))

size_train = int(RATIO_TRAIN * len(file_names))
size_valid = int(RATIO_VALID * len(file_names))

random.shuffle(file_names)
train = file_names[:size_train]
valid = file_names[size_train:size_train + size_valid]
test = file_names[size_train + size_valid:]

print("Resulting split: #train %d, #valid %d, #test %d" % (len(train), len(valid), len(test)))
print("Resulting ratio: train %.2f, valid %.2f, test %.2f"
      % (len(train) / len(file_names), len(valid) / len(file_names), len(test) / len(file_names)))

with open(os.path.join(FOLDER_SETS, SET, "splits", 'train.txt'), 'w+') as f:
    for item in train:
        f.write(item + '\n')

with open(os.path.join(FOLDER_SETS, SET, "splits", 'test.txt'), 'w+') as f:
    for item in test:
        f.write(item + '\n')

with open(os.path.join(FOLDER_SETS, SET, "splits", 'valid.txt'), 'w+') as f:
    for item in valid:
        f.write(item + '\n')
