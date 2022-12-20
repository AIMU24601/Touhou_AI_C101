import ctypes
import time
import sys
import pynput
import os
import random

import numpy as np
import cv2
import win32gui

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pfrl

from PIL import ImageGrab
from tqdm import tqdm

WINDOW_NAME = "東方妖々夢　～ Perfect Cherry Blossom. ver 1.00"

SENDINPUT = ctypes.windll.user32.SendInput

global device
device = torch.device("cuda")

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

class Qfunction(torch.nn.Module):
    def __init__(self):
        print("Initializing DQN...")
        print("Model Building")
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, 3)
        self.conv2 = nn.Conv2d(16, 32, 5, 2, 2)
        self.conv3 = nn.Conv2d(32, 64, 5, 2, 2)
        self.l4 = nn.Linear(320, 5) #アクション数は5通り #320の部分を変更？

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h3 = h3.view(h3.size(0), -1)
        return pfrl.action_value.DiscreteActionValue(self.l4(h3))

def random_action():
    return np.random.randint(0, 5)

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

def screen_shot(model):
    """画面を取得し入力とテンプレートマッチングに使う画像を返す関数

    Returns:
        output: 入力用画像
        img_template: テンプレートマッチング用画像
    """
    handle = win32gui.FindWindow(None, WINDOW_NAME)
    rect = win32gui.GetWindowRect(handle)

    grabed_image = ImageGrab.grab()
    croped_image = grabed_image.crop(rect)
    croped_image = croped_image.crop((35, 42, 420, 490))

    #croped_image.save("screenshot.jpg")

    img = np.asarray(croped_image, dtype="uint8")
    #テンプレートマッチング用
    img_template = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #入力用画像
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, dsize=(128, 160))
    mean = [90.9, 72.69, 86.76]
    std = [73.8, 70.29, 70.12]
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    img = t(img)
    img = img.unsqueeze(0)
    model.to(device); img=img.to(device)
    output = model(img)
    output = torch.argmax(output, dim=1)
    output = output.cpu().squeeze(0)
    output = output.detach().numpy()
    #cv2.imwrite("test.jpg", img=img)

    return output, img_template

def deathcheck(img, img_d):
    match_result_d = cv2.matchTemplate(img, img_d, cv2.TM_CCOEFF_NORMED)
    death_check = np.column_stack(np.where(match_result_d >= 0.75))

    return death_check

def score(img, img_p, img_t, time_hit, previous_p_check, previous_t_check):
    reward = 0
    match_result_p = cv2.matchTemplate(img, img_p, cv2.TM_CCOEFF_NORMED)
    match_result_t = cv2.matchTemplate(img, img_t, cv2.TM_CCOEFF_NORMED)
    #match_result_c = cv2.matchTemplate(img, img_c, cv2.TM_CCOEFF_NORMED)
    p_check = np.column_stack(np.where(match_result_p >= 0.5))
    t_check = np.column_stack(np.where(match_result_t >= 0.7))
    #c_check = np.column_stack(np.where(match_result_c >= 0.8))
    if time.time() - time_hit > 3:
        if len(p_check) >= 1:
            print("The number of P: {}".format(len(p_check)))
            reward += max(len(p_check)-previous_p_check, 0)
            previous_p_check = len(p_check)
        if len(t_check) >= 1:
            print("The number of Ten: {}".format(len(t_check)))
            reward += max(len(t_check)-previous_t_check, 0)
            previous_t_check = len(t_check)
        """if len(c_check) >= 1:
            print("Chapter finished")
            cool_time = time.time()
            if cool_time - previous_time > 5:
                reward += 100
                previous_time = cool_time
        """
    return reward, previous_p_check, previous_t_check

def retry():
    releasekey(0x2c)#releaseZ
    time.sleep(1)
    Retry_check = False
    while Retry_check:
        Retry_check = RetryCheck()
        presskey(0x2c)
        time.sleep(1/60)
        releasekey(0x2c)
        time.sleep(1)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)
    time.sleep(1)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)#コンテニューでいいえ
    #print("いいえ")
    time.sleep(3)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)#あなたの腕前から次へ
    #print("腕前")
    time.sleep(90/60)
    presskey(0xcd)#RIGHT
    time.sleep(1/60)
    releasekey(0xcd)#RIGHT
    time.sleep(10/60)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)#リプレイ保存でいいえ
    #print("リプレイ")
    time.sleep(2)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)#Start
    #print("Start")
    time.sleep(90/60)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)#入門を押す
    #print("入門")
    time.sleep(90/60)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)#自機選択
    #print("自機選択")
    time.sleep(90/60)
    presskey(0x2c)
    time.sleep(1/60)
    releasekey(0x2c)#装備選択
    #print("装備選択")
    time.sleep(1)
    #ここまでリトライ処理

    return None

def RetryCheck():
    #あなたの腕前の画面に遷移しているかの確認
    RETRYCHECKFILE = "images/udemae.png"
    IMG_R = cv2.imread(RETRYCHECKFILE, cv2.IMREAD_COLOR)
    X = 284
    Y = 290
    W = 670
    H = 740 #撮影の座標指定
    img = ImageGrab.grab((X, Y, W, H))
    img = np.asarray(img, dtype="uint8")
    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    match_result_r = cv2.matchTemplate(img_1, IMG_R, cv2.TM_CCOEFF_NORMED)
    retry_check = np.column_stack(np.where(match_result_r >= 0.75))
    if len(retry_check) >= 1:
        return True
    else:
        return False

def main():

    #ハイパーパラメータ
    GANMA = 0.99 #割引率
    NUM_EPISODE = 10000 #総試行回数
    MAX_EPISODE_LEN = 1000
    BATCH_SIZE = 32
    CAPACITY = 10000

    #画像処理用
    DEATHCHECK_FILE = "images/continue.png"
    POWER = "images/p.png"
    TEN = "images/t.png"
    #CHAPER =
    IMG_D = cv2.imread(DEATHCHECK_FILE, cv2.IMREAD_COLOR) #被弾確認用
    IMG_P = cv2.imread(POWER, cv2.IMREAD_COLOR) #P確認用
    IMG_T = cv2.imread(TEN, cv2.IMREAD_COLOR) #点確認用
    #IMG_C = cv2.imread(CHAPTER, cv2.IMREAD_COLOR) #Chapter Finish確認用

    #DQNのセットアップ
    q_func = Qfunction()
    #q_func.to_gpu(0)
    optimizer = torch,optim.Adadelta(q_func.parameters(), rho=0.95, eps=1e-06)
    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.0, decay_steps=NUM_EPISODE*100, random_action_func=random_action)
    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = pfrl.agents.DQN(q_func, optimizer, replay_buffer, GANMA, explorer, gpu=0, replay_start_size=5000, minibatch_size=100, update_interval=50, target_update_interval=2000, phi=phi)
    #agent.load("agent_TouhouAIDDQN_3000")

    try:
        model = torch.load("model/Unet-_mIoU-back-0.537.pt")

        command = 0
        reward = 0
        time_step = 0
        previous_p_check = 0
        previous_t_check = 0
        time.sleep(1)
        commandstart(-1)
        time.sleep(1/60)
        commandend(-1)
        for episode in range(1, NUM_EPISODE+1):
            presskey(0x2c) #pressZ
            print("episode: {}".format(episode))
            done = False
            reset = False
            r = 0
            t = 0
            time_hit = time.time()
            obs, _ = screen_shot(model)
            time.sleep(1)
            while True:
                print(time_step)
                command = agent.act(obs)
                #print(command)
                commandstart(command)
                obs, img_template = screen_shot(model)
                death = deathcheck(img_template, IMG_D)
                reset = t == MAX_EPISODE_LEN
                if len(death) >= 1:
                    done = True
                    print("被弾")
                    r = -100
                    print("reward: {}".format(r))
                elif reset:
                    pass
                else:
                    r, previous_p_check,previous_t_check = score(img_template, IMG_P, IMG_T, time_hit, previous_p_check,previous_t_check)
                    print("reward: {}".format(r))
                    agent.observe(obs, r, done, reset)
                commandend(command)
                if done or reset:
                    time.sleep(1)
                    retry()
                    time.sleep(1)
                    break
                t += 1
            previous_p_check = 0
            previous_t_check = 0
            time_step += t
            print("time step: {}, average time step: {}".format(t, time_step/episode))
            if episode % 1000 == 0: #50エピソード毎にエージェントモデルを保存
                print("model saved")
                agent.save("agent_TouhouAIDQN_" + str(episode))
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()