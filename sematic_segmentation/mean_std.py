#calculate mean and std of own datasets for normalization.

import os

import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = ("E:/touhou/resize/with_background/")

train_dir = os.path.join(DATA_DIR, "train/7")

valid_dir = os.path.join(DATA_DIR, "val/7")

test_dir = os.path.join(DATA_DIR, "test/7")

img = list()
m = np.zeros((160,120,3))

x_train_ids = os.listdir(train_dir)
x_valid_ids= os.listdir(valid_dir)
x_test_ids = os.listdir(test_dir)

train_d = [os.path.join(train_dir, img_id) for img_id in x_train_ids]
valid_d = [os.path.join(valid_dir, img_id) for img_id in x_valid_ids]
test_d = [os.path.join(test_dir, img_id) for img_id in x_test_ids]

for i in tqdm(range(len(train_d))):
    img.append(cv2.resize(cv2.imread(train_d[i]), dsize=(120,160)))
for i in range(len(valid_d)):
    img.append(cv2.resize(cv2.imread(valid_d[i]), dsize=(120,160)))
for i in range(len(test_d)):
    img.append(cv2.resize(cv2.imread(test_d[i]), dsize=(120,160)))

img = np.array(img)

"""
for i in img:
    m += i
me = m/(len(train_d)+len(valid_d)+len(test_d))
mea = np.sum(me,axis=0)
mean = np.sum(mea,axis=0)
mean = mean/(160*120)

print(mean)
"""
# mean [17.41039304 17.81785532 18.53550618] BGRなので注意
# mean [18.5, 17.8, 17.4] RGB
mean = [86.75943533, 72.68530017, 90.89596181]
#[86.75943533 72.68530017 90.89596181] 背景込みの平均
#[90.9, 72.69, 86.76] RGB

tmp = np.stack([mean for _ in range(120)], axis=0)
tmp_2 = np.stack([tmp for _ in range(160)], axis=0)
tmp_3 = np.stack([tmp_2 for _ in range(len(train_d)+len(valid_d)+len(test_d))], axis=0)

img = img - tmp_3
img = img ** 2

img = img/(len(train_d)+len(valid_d)+len(test_d))
s = np.sum(np.sum(np.sum(img,axis=0),axis=0),axis=0)
var= s/(160*120)
print(var)

# var [3024.09503826 3103.64208278 3277.34470531] BGRなので注意
#[4916.6138056  4941.06618017 5447.14941674]
std = var ** (1/2)
print(std)
# std [54.99177246 55.7103409  57.24809783] BGRなので注意
# std [57.3, 55.7, 55.0] RGB
# std [70.11856962 70.29271783 73.80480619] 背景込みの標準偏差(BGR)
#[73.8, 70.29, 70.12] RGB