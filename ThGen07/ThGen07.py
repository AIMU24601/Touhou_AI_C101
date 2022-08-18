from random import random
import cv2
from PIL import Image
import random
import numpy as np
import math
import os

number_player = 1
min_number_enemy = 1
max_number_enemy = 10
min_number_bullet = 1
max_munber_bullet = 100
MIN_NUMBER_BG_PARTS = 1
MAX_NUMBER_BG_PATRS = 10
player_list = list()
enemy_list = list()
bullet_list = list()
bg_list = list()
bg_pars_list = list()

DATA_DIR = "E:/touhou/data/7/"

bg_dir = os.path.join(DATA_DIR, "bg_tmp/")
bg_parts_dir = os.path.join(DATA_DIR, "bg_parts_tmp/")

#貼り付ける画像をロード
for i in range(1, 23):
    tmp = Image.open("E:/touhou/data/7/player_tmp/" + str(i) + ".png")
    player_list.append(tmp)

for i in range(1, 375):
    tmp = Image.open("E:/touhou/data/7/enemy_tmp/" + str(i) + ".png")
    enemy_list.append(tmp)

for i in range(1, 173):
    tmp = Image.open("E:/touhou/data/7/bullet_tmp/" + str(i) + ".png")
    bullet_list.append(tmp)

ids = os.listdir(bg_dir)
image_ids = [os.path.join(bg_dir,image_id) for image_id in ids]

for i in range(len(image_ids)):
    bg_list.append(Image.open(image_ids[i]))

ids = os.listdir(bg_parts_dir)
image_ids = [os.path.join(bg_parts_dir,image_id) for image_id in ids]

for i in range(len(image_ids)):
    bg_pars_list.append(Image.open(image_ids[i]))

#ゲームのスクリーンショットにサイズを合わせる→160*120にするための比を求める
img = cv2.imread("../screenshot.jpg")
target_x = 128
target_y = 160
row = len(img)
row_ratio = target_y/row
column = len(img[0])
column_raio = target_x/column

def ThGen(t):
    #画像を貼り付ける所を作成
    train = np.zeros((target_y, target_x))
    bg = np.zeros((target_y, target_x))
    annot = np.zeros((target_y, target_x)) # 0 means this pixel does not belong to any classes
    #ndarrayはcv2で使うものなのでPILに変換する
    train = Image.fromarray(train,"RGBA")
    bg = Image.fromarray(bg,"RGBA")

    #背景用の画像を生成
    b = 1
    for i in range(1):
        tmp = random.randint(0,len(bg_list)-1) #貼り付ける背景を決定
        im = bg_list[tmp]
        bg.paste(im)

    b_parts = random.randint(MIN_NUMBER_BG_PARTS,MAX_NUMBER_BG_PATRS)
    for i in range(b):
        x = random.randint(0, target_x-1)
        y = random.randint(0,target_y-1)
        tmp = random.randint(0,len(bg_pars_list)-1)
        im = bg_pars_list[tmp]
        bg.paste(im,(x,y),mask=im)

    e = random.randint(min_number_enemy, max_number_enemy)
    for i in range(e):
        x = random.randint(0, target_x-1)
        y = random.randint(0, target_y-1)
        tmp = random.randint(0,len(enemy_list)-1)
        im = enemy_list[tmp]
        w, h = im.size
        w = math.ceil(w*column_raio)
        h = math.ceil(h*row_ratio)
        im = im.resize((w,h))
        train.paste(im, (x, y), mask=im)
        for j in range(y, y+h):
            for k in range(x, x+w):
                if j < target_y and k < target_x:
                    tmp = train.getpixel((k, j))
                    if sum(tmp) >= (120+255):
                        annot[j][k] = 2 #one-hot of enemy

    b = random.randint(min_number_bullet, max_munber_bullet)
    for i in range(b):
        x = random.randint(0, target_x-1)
        y = random.randint(0, target_y-1)
        tmp = random.randint(0, 171)
        im = bullet_list[tmp]
        w, h = im.size
        w = math.ceil(w*column_raio)
        h = math.ceil(h*row_ratio)
        im = im.resize((w,h))
        im = im.rotate(random.randint(0, 360))
        train.paste(im, (x, y), mask=im)
        for j in range(y, y+h):
            for k in range(x, x+w):
                if j < target_y and k < target_x:
                    tmp = train.getpixel((k, j))
                    if sum(tmp) >= (120+255):
                        annot[j][k] = 3 #one-hot of bullet

    p = random.randint(number_player, number_player)
    for i in range(p):
        #貼り付ける座標を決定
        x = random.randint(0, target_x-1)
        y = random.randint(0, target_y-1)
        #貼り付ける画像を決定
        tmp = random.randint(0, 21)
        im = player_list[tmp]
        w, h = im.size
        w = math.ceil(w*column_raio)
        h = math.ceil(h*row_ratio)
        im = im.resize((w,h))
        train.paste(im, (x, y), mask=im)
        for j in range(y, y+h):
            for k in range(x, x+w):
                if j < target_y and k < target_x:
                    tmp = train.getpixel((k, j))
                    if sum(tmp) >= (120+255):
                        annot[j][k] = 1 #one-hot of player

    #train.save("E:/touhou/resize/train/7/" + str(t) + ".png")
    np.save("E:/touhou/resize/with_background/train_annot/7/" + str(t), annot)

    eval_save = np.zeros((target_y, target_x, 3))
    for j in range(target_y):
            for k in range(target_x):
                if annot[j][k] == 1: # this pixel belongs to player class
                    eval_save[j][k] = [255, 0, 0]
                elif annot[j][k] == 2: # this pixel belongs to enemy class
                    eval_save[j][k] = [0,255, 0]
                elif annot[j][k] == 3: # this pixel belongs to bullet class
                    eval_save[j][k] = [0, 0, 255]
    cv2.imwrite("E:/touhou/resize/with_background//annot_img/7/" + str(t) + ".png", eval_save)

    bg.paste(train,mask=train)
    bg.save("E:/touhou/resize/with_background/train/7/"+str(t)+".png")

if __name__ == "__main__":
    for i in range(0, 20):
        ThGen(i)
        if i %100 == 0:
            print(i)
