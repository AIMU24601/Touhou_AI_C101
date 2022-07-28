from random import random
import cv2
from PIL import Image
import random
import numpy as np

number_player = 1
min_number_enemy = 1
max_number_enemy = 10
min_number_bullet = 1
max_munber_bullet = 100
player_list = list()
enemy_list = list()
bullet_list = list()

img = cv2.imread("C:/touhou/data/7/enemy/10.png")
print(type(img))
print(img.shape)
for i in range(len(img)):
    for j in range(len(img[0])):
        pass
        #print(img[i][j])
print(img.sum(axis=2))
img = cv2.imread("screenshot.jpg")
row = len(img)
column = len(img[0])

#画像を貼り付ける所を作成
train = np.zeros((row, column))
train = train + 31
#ndarrayはcv2で使うものなのでPILに変換する
train = Image.fromarray(train, "RGBA")
eval = train

im = Image.open("C:/touhou/data/7/player/1.png")

p = random.randint(number_player, number_player)
for i in range(number_player):
    #貼り付ける画像を決定
    """
    player = random.randint(0, len(player_list)-1)
    player_img = player_list[player]
    """
    #貼り付ける座標を決定
    x = random.randint(0, column-1)
    y = random.randint(0, row-1)
    train.paste(im, (x, y))
    print(im)
    print(train)
    train.save("C:/touhou/train/7/1.png")

e = random.randint(min_number_enemy, max_number_enemy)
for i in range(e):
    pass

b = random.randint(min_number_bullet, max_munber_bullet)
for i in range(b):
    pass
