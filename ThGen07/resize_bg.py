import os

import cv2
import numpy as np

DATA_DIR = "E:/touhou/data/7/bg_parts/"

ids = os.listdir(DATA_DIR)

image_ids = [os.path.join(DATA_DIR,image_id) for image_id in ids]

for k in range(len(image_ids)):
    img = cv2.imread(image_ids[k])
    #img = cv2.resize(img,dsize=(128,160))
    row = img.shape[0]
    column = img.shape[1]

    ch_a = np.zeros((row, column))
    ch_a = ch_a.astype(np.uint8)

    ch_b, ch_g, ch_r = cv2.split(img[:,:,:3])
    dst = cv2.merge((ch_b, ch_g, ch_r, ch_a))

    for i in range(row):
        for j in range(column):
            if sum(dst[i][j]) >= 100:
                dst[i][j][-1] = 255
    dst = cv2.resize(dst,dsize=None,fx=0.6,fy=0.6)
    cv2.imwrite("E:/touhou/data/7/bg_parts_tmp/"+str(k)+".png",dst)