import os
import sys
from tkinter import image_names

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

DATA_DIR = "E:/touhou/resize/with_background/"

x_train_dir = os.path.join(DATA_DIR, "train/7")
y_train_dir = os.path.join(DATA_DIR, "train_annot/7")

x_valid_dir = os.path.join(DATA_DIR, "val/7")
y_valid_dir = os.path.join(DATA_DIR, "val_annot/7")

global CLASSES
CLASSES = ["background","player", "enemy", "bullet"]
#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(edgeitems=1000000)

class Dataset(BaseDataset):
    #mean = [90.9, 72.69, 86.76]
    mean = np.mean([90.9, 72.69, 86.76])
    #std = [73.8, 70.29, 70.12]
    std = np.mean([73.8, 70.29, 70.12]) #グレースケール用

    CLASSES = ["background","player", "enemy", "bullet"]

    def __init__(self, images_dir, masks_dir, classes=None, preprocessing=None):
        self.image_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.preprocessing = preprocessing

    def __getitem__(self, i):
        #image = cv2.copyMakeBorder(cv2.imread(self.images_fps[i]),0,0,0,8,cv2.BORDER_CONSTANT,value=(0,0,0)) #(160,128) 32で割り切れるように右をゼロパディング
        image = cv2.imread(self.images_fps[i])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #グレースケール用
        #mask = np.pad(np.load(self.masks_fps[i]),((0,0),(0,8))) #画像サイズに合わせてパディング
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

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=len(CLASSES)):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve= 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            output = model(image)
            loss = criterion(output, mask)
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))

            if min_loss > (test_loss/len(val_loader)):
                print("Loss Decreasing.. {:.3f} >> {:.3f}".format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                if decrease % 1 == 0:
                    print("saving model...")
                    torch.save(model, 'model/gray/Unet-_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader))) #Train途中もモデルを保存するときは実行する

            if (test_loss/len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(val_loader))
                print(f"Loss Not Decrease for {not_improve} time")
                if not_improve == 20:
                    print("Loss not decrease for 20 times, Stop Training")
                    break

            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
            "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
            "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
            "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
            "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
            "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
            "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
            "Time: {:.2f}m".format((time.time()-since)/60))

    history = {'train_loss' : train_losses, 'val_loss': test_losses,
        'train_miou' :train_iou, 'val_miou':val_iou,
        'train_acc' :train_acc, 'val_acc':val_acc, 'lrs': lrs}
    print("Total time: {:.2f} m".format((time.time() - fit_time)/60))
    return history

def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

if __name__ == "__main__":
    """
    dataset = Dataset(images_dir=x_train_dir, masks_dir=y_train_dir, classes=["player", "enemy", "bullet"])
    print(dataset)

    image, mask = dataset[0]
    visualize(image=image.permute(1, 2, 0), mask=mask)
    """

    print("Model Setup...")
    #model = smp.Unet("efficientnet-b4", encoder_weights="imagenet", classes=len(CLASSES), activation=None)
    model = smp.Unet("efficientnet-b4", encoder_weights="imagenet", classes=len(CLASSES), activation=None, in_channels=1) #グレースケール用
    print("Done")

    print("Dataset Loading...")
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES)
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("Done")

    max_lr = 1e-3
    epoch = 100
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    #criterion =
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

    history = fit(epoch, model, train_loader, valid_loader, criterion, optimizer, sched)
    np.save("history_with_back.npy", history)

    plot_loss(history)
    plot_score(history)
    plot_acc(history)
