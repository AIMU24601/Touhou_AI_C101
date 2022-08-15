#calculate mean and std of own datasets for normalization.

import os

import cv2
import numpy as np

DATA_DIR = ("E:/touhou/")

train_dir = os.path.join(DATA_DIR, "train/7")

valid_dir = os.path.join(DATA_DIR, "val/7")

test_dir = os.path.join(DATA_DIR, "test/7")

img = list()
#mean = np.zeros((160,120,3))

x_train_ids = os.listdir(train_dir)
x_valid_ids= os.listdir(valid_dir)
x_test_ids = os.listdir(test_dir)

train_d = [os.path.join(train_dir, img_id) for img_id in x_train_ids]
valid_d = [os.path.join(valid_dir, img_id) for img_id in x_valid_ids]
test_d = [os.path.join(test_dir, img_id) for img_id in x_test_ids]

"""
for i in range(len(train_d)):
    img.append(cv2.resize(cv2.imread(train_d[i]), dsize=(160,120)))
    print(i)
for i in range(len(valid_d)):
    img.append(cv2.resize(cv2.imread(valid_d[i]), dsize=(160,120)))
for i in range(len(test_d)):
    img.append(cv2.resize(cv2.imread(test_d[i]), dsize=(160,120)))
"""

"""
for i in range(len(train_d)):
    img.append(cv2.resize(cv2.imread(train_d[i]), dsize=(120,160)))
    print(i)
for i in range(len(valid_d)):
    img.append(cv2.resize(cv2.imread(valid_d[i]), dsize=(120,160)))
    print(i)
for i in range(len(test_d)):
    img.append(cv2.resize(cv2.imread(test_d[i]), dsize=(120,160)))
    print(i)


img = np.array(img)
print(img.shape)
"""
"""
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img)
print(img.shape)
"""

"""
for i in img:
    mean += i
m = mean/(len(train_d)+len(valid_d)+len(test_d))
me = np.sum(m,axis=0)
mea = np.sum(me,axis=0)
mea = mea/(160*120)
"""

#print(mea)
# mean [17.41039304 17.81785532 18.53550618] BGRなので注意
# mean [18.5, 17.8, 17.4] RGB

for i in range(3):
    img.append(cv2.resize(cv2.imread(train_d[i]), dsize=(120,160)))
    print(i)

img = np.array(img)
print(img.shape)

mean = np.array([17.41039304, 17.81785532, 18.53550618])
print(mean.shape)
tmp = np.stack([mean for _ in range(120)], axis=0)
print(tmp.shape)
tmp_2 = np.stack([tmp for _ in range(160)], axis=0)
print(tmp_2.shape)
tmp_3 = np.stack([tmp_2 for _ in range(3)], axis=0)
print(tmp_3.shape)

img = img - tmp_3
img = img ** 2
print(img.shape)

img = img/(len(train_d)+len(valid_d)+len(test_d))
s = np.sum(np.sum(np.sum(img,axis=0),axis=0),axis=0)
std = s/(160*120)
print(std.shape)
print(std)
# std [0.35001241 0.36715648 0.39789724] BGRなので注意
# std [0.398, 0.367, 0.350] RGB