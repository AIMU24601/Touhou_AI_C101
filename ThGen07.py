from random import random
import cv2
from PIL import Image
import random
from cv2 import imwrite
import numpy as np

number_player = 1
min_number_enemy = 1
max_number_enemy = 10
min_number_bullet = 1
max_munber_bullet = 100
player_list = list()
enemy_list = list()
bullet_list = list()

#貼り付ける画像をロード
for i in range(1, 23):
    tmp = Image.open("C:/touhou/data/7/player/" + str(i) + ".png")
    player_list.append(tmp)

for i in range(1, 375):
    tmp = Image.open("C:/touhou/data/7/enemy/" + str(i) + ".png")
    enemy_list.append(tmp)

for i in range(1, 173):
    tmp = Image.open("C:/touhou/data/7/bullet/" + str(i) + ".png")
    bullet_list.append(tmp)
print(len(bullet_list))

#ゲームのスクリーンショットにサイズを合わせる
img = cv2.imread("screenshot.jpg")
row = len(img)
column = len(img[0])

#画像を貼り付ける所を作成
train = np.zeros((row, column))
train = train + 31
eval = np.zeros((row, column, 4))
#ndarrayはcv2で使うものなのでPILに変換する
train = Image.fromarray(train, "RGBA")

p = random.randint(number_player, number_player)
for i in range(p):
    #貼り付ける画像を決定
    """
    player = random.randint(0, len(player_list)-1)
    player_img = player_list[player]
    """
    #貼り付ける座標を決定
    x = random.randint(0, column-1)
    y = random.randint(0, row-1)
    #貼り付ける画像を決定
    tmp = random.randint(0, 21)
    im = player_list[tmp]
    train.paste(im, (x, y))
    w, h = im.size
    for j in range(y, y+h):
        for k in range(x, x+w):
            if j < row and k < column:
                tmp = train.getpixel((k, j))
                if tmp[0] != 31:
                    eval[j][k] = [1, 0, 0, 0]

e = random.randint(min_number_enemy, max_number_enemy)
for i in range(e):
    x = random.randint(0, column-1)
    y = random.randint(0, row-1)
    tmp = random.randint(0, 373)
    im = enemy_list[tmp]
    train.paste(im, (x, y))
    w, h = im.size
    for j in range(y, y+h):
        for k in range(x, x+w):
            if j < row and k < column:
                tmp = train.getpixel((k, j))
                if tmp[0] != 31:
                    eval[j][k] = [0, 1, 0, 0]

b = random.randint(min_number_bullet, max_munber_bullet)
for i in range(b):
    x = random.randint(0, column-1)
    y = random.randint(0, row-1)
    tmp = random.randint(0, 171)
    print(tmp)
    im = bullet_list[tmp]
    im = im.rotate(random.randint(0, 360))
    train.paste(im, (x, y))
    w, h = im.size
    for j in range(y, y+h):
        for k in range(x, x+w):
            if j < row and k < column:
                tmp = train.getpixel((k, j))
                if tmp[0] != 31:
                    eval[j][k] = [0, 0, 1, 0]

train.save("C:/touhou/train/7/1.png")

eval_save = np.zeros((row, column, 3))
for j in range(row):
        for k in range(column):
            if eval[j][k][0] == 1:
                eval_save[j][k] = [255, 0, 0]
            if eval[j][k][1] == 1:
                eval_save[j][k] = [0,255, 0]
            if eval[j][k][2] == 1:
                eval_save[j][k] = [0, 0, 255]
cv2.imwrite("C:/touhou/eval/7/1.png", eval_save)