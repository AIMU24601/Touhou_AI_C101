from cProfile import label
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 100
num_baatch = 5
learning_rate = 0.01
image_size = 385*448

DATA_DIR = "E:/touhou/"

x_train_dir = os.path.join(DATA_DIR, "train/7")
y_train_dir = os.path.join(DATA_DIR, "train_annot/7")

x_valid_dir = os.path.join(DATA_DIR, "val/7")
y_valid_dir = os.path.join(DATA_DIR, "val_annot/7")

x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "test_annot/7")

global CLASSES
CLASSES = ["player", "enemy", "bullet"]

class Dataset(BaseDataset):
    mean = [18.5, 17.8, 17.4]
    std = [57.3, 55.7, 55.0]

    CLASSES = ["player", "enemy", "bullet"]

    def __init__(self, images_dir, masks_dir, classes=None, preprocessing=None):
        self.image_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.masks_fps[i])
        """
        mask[:,:,:3] = mask[:,:,:3]*255
        mask[:,:,3] = mask[:,:,3]+200
        """

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample(mask)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        image = t(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.image_ids)

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        x =self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

if __name__ == "__main__":
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES)
    #valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    #valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = Net(image_size, output_size=image_size).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        loss_sum = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            print(inputs.size())

            optimizer.zero_grad()

            inputs = inputs.view(-1, image_size)
            outputs = model(inputs)
            print(inputs.size())
            print(labels.size())
            labels = labels.view(-1, image_size)
            print(labels.size())

            loss = criterion(outputs, labels)
            loss_sum += loss

            loss.backward()

            optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epochs} Loss: {loss_sum.item() / len(train_loader)}")

    torch.save(model, "touhou.pt")
