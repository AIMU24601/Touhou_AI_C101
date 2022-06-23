import ctypes
import time
import sys
import numpy as np
import cv2
import pynput
import win32gui

from PIL import ImageGrab

WINDOW_NAME = "東方妖々夢　～ Perfect Cherry Blossom. ver 1.00"

SENDINPUT = ctypes.windll.user32.SendInput

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wvk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlagss", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_ulong),
                ("dy", ctypes.c_ulong),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def presskey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.ki = pynput._util.win32.KEYBDINPUT(0, hexKeyCode, 0x0008, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(1), ii_)
    SENDINPUT(1, ctypes.pointer(x), ctypes.sizeof(x))

def releasekey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.ki = pynput._util.win32.KEYBDINPUT(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(1), ii_)
    SENDINPUT(1, ctypes.pointer(x), ctypes.sizeof(x))

def commandstart(action):
    if action == -1:
        presskey(0x01)#ESC
    elif action == 0:
        #presskey(0x2c)#pressZ
        pass
    elif action == 1:
        presskey(0xcb)#LEFT
    elif action == 2:
        presskey(0xc8)#UP
    elif action == 3:
        presskey(0xcd)#RIGHT
    elif action == 4:
        presskey(0xd0)#DOWN

def commandend(action):
    if action == -1:
        releasekey(0x01)#ESC
    elif action == 0:
        #releasekey(0x2c)#Z
        pass
    elif action == 1:
        releasekey(0xcb)#LEFT
    elif action == 2:
        releasekey(0xc8)#UP
    elif action == 3:
        releasekey(0xcd)#RIGHT
    elif action == 4:
        releasekey(0xd0)#DOWN

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

def main():
    try:
        state, template = screen_shot()
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()