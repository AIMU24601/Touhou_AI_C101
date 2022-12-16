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

from PIL import ImageGrab
from tqdm import tqdm


from collections import namedtuple
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

WINDOW_NAME = "東方妖々夢　～ Perfect Cherry Blossom. ver 1.00"

#ハイパーパラメータ
GANMA = 0.99 #割引率
NUM_EPISODE = 10000 #総試行回数
BATCH_SIZE = 32
CAPACITY = 10000

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

class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY #メモリの最大長さ
        self.memory = [] #経験を保存する変数
        self.index = 0 #保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Brain:

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(num_states, 32))
        self.model.add_module("relu1", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(32, 32))
        self.model.add_module("relu2", nn.ReLU())
        self.model.add_module("fc3", nn.Linear(32, num_actions))

        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s:s is not None, batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch+GANMA*next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5+(1/(episode+1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)

        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

class Agent:

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q(self):
        self.brain.replya()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

class Environment:

    def __init__(self):
        self.env = 
        self.num_states = 
        self.num_actions = 

        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        episode_final = Falseframes = []

        for episode in range(NUM_EPISODE):
            observation = 
            state = observation
            state = torch.

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
        command = 0
        reward = 0
        time_step = 0
        presskey(0x2c) #pressZ
        for episode in range(1, NUM_EPISODE+1):
            time.sleep(1)
            commandstart(-1)
            time.sleep(1/60)
            commandend(-1)
            print("episode: {}".format(episode))
            done = False
            r = 0
            t = 0
            time.sllep(1)
            while not done:
                img, img_template = screen_shot()
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()