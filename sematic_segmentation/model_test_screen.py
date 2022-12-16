import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms as T
import torchvision
import torch.nn.functional  as F
from torch.autograd import Variable

from PIL import Image
from tqdm import tqdm

import time
import segmentation_models_pytorch as smp

DATA_DIR = "E:/touhou/resize/"

x_test_dir = os.path.join(DATA_DIR, "test/7")
y_test_dir = os.path.join(DATA_DIR, "test_annot/7")

global CLASSES
CLASSES = ["background","player", "enemy", "bullet"]
global device
device = torch.device("cuda")

if __name__ =="__main__":
    model = torch.load("Unet-_mIoU-back-0.537.pt")
    image = cv2.imread("screenshot.jpg")
    image = cv2.resize(image,dsize=(128,160))
    #image = cv2.copyMakeBorder(image,0,0,0,8,cv2.BORDER_CONSTANT,value=(0,0,0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean = [90.9, 72.69, 86.76]
    std = [73.8, 70.29, 70.12]
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    image = image.unsqueeze(0)
    model.to(device); image=image.to(device)
    output = model(image)
    output = torch.argmax(output, dim=1)
    output = output.cpu().squeeze(0)
    output = output.detach().numpy()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    image = cv2.imread("screenshot.jpg")
    image = cv2.resize(image,dsize=(128,160))
    #image = cv2.copyMakeBorder(image,0,0,0,8,cv2.BORDER_CONSTANT,value=(0,0,0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image)
    ax2.imshow(output)
    #plt.imshow(output)
    plt.show()