from pickletools import uint8
import cv2
import numpy as np

bullet_list = list()

for i in range(1, 23):
    tmp = cv2.imread("C:/touhou/data/7/player/" + str(i) + ".png")
    bullet_list.append(tmp)

for k in range(0, 22):
    img = bullet_list[k]
    row = img.shape[0]
    column = img.shape[1]

    ch_a = np.zeros((row, column))
    ch_a = ch_a.astype(np.uint8)

    ch_b, ch_g, ch_r = cv2.split(img[:,:,:3])
    dst = cv2.merge((ch_b, ch_g, ch_r, ch_a))

    for i in range(row):
        for j in range(column):
            if sum(dst[i][j]) >= 120:
                dst[i][j][-1] = 255

    cv2.imwrite("C:/touhou/data/7/player_tmp/" + str(k+1) + ".png", dst)