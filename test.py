import ctypes
import time
import sys
import numpy as np
import cv2
import pynput
import win32gui

from PIL import ImageGrab

WINDOW_NAME = "東方妖々夢　～ Perfect Cherry Blossom. ver 1.00"

def screen_shot():
    """画面を取得し入力とテンプレートマッチングに使う画像を返す関数

    Returns:
        img: 入力用画像
        img_template: テンプレートマッチング用画像
    """
    handle = win32gui.FindWindow(None, WINDOW_NAME)
    rect = win32gui.GetWindowRect(handle)

    grabed_image = ImageGrab.grab()
    croped_image = grabed_image.crop(rect)
    croped_image = croped_image.crop((35, 42, 420, 490))

    croped_image.save("screenshot.jpg")

    img = np.asarray(croped_image, dtype="uint8")
    #テンプレートマッチング用
    img_template = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #入力用画像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, dsize=(120, 160))
    cv2.imwrite("test.jpg", img=img)

    return img, img_template

screen_shot()