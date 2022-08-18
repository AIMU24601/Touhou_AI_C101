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

DATA_DIR = "E:/touhou/resize/with_background/"

x_test_dir = os.path.join(DATA_DIR, "test/7")
y_test_dir = os.path.join(DATA_DIR, "test_annot/7")

global CLASSES
CLASSES = ["background","player", "enemy", "bullet"]
global device
device = torch.device("cuda")

class testDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ["background","player", "enemy", "bullet"]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        #image = cv2.copyMakeBorder(cv2.imread(self.images_fps[i]),0,0,0,8,cv2.BORDER_CONSTANT,value=(0,0,0)) #(160,128) 32で割り切れるように右をゼロパディング
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = np.pad(np.load(self.masks_fps[i]),((0,0),(0,8))) #画像サイズに合わせてパディング
        mask = np.load(self.masks_fps[i])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        #t = T.Compose([T.ToTensor()])
        #image = t(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.ids)

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

def predict_image_mask_miou(model, image, mask):
    mean = [18.5, 17.8, 17.4]
    std = [57.3, 55.7, 55.0]
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def predict_image_mask_pixel(model, image, mask):
    mean = [18.5, 17.8, 17.4]
    std = [57.3, 55.7, 55.0]
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc

def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou

def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy

if __name__ == "__main__":
    # create test dataset
    test_dataset = testDataset(
    x_test_dir,
    y_test_dir,
    #preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

    print("Model Loading...")
    model = torch.load("Unet-_mIoU-back-0.493.pt")
    print("Done")
    test_dataloader = DataLoader(test_dataset)

    mob_miou = miou_score(model, test_dataset)
    print('Test Set mIoU', np.mean(mob_miou))

    mob_acc = pixel_acc(model, test_dataset)
    print('Test Set Pixel Accuracy', np.mean(mob_acc))

    image2, mask2 = test_dataset[15]
    pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
    ax1.imshow(image2)
    ax1.set_title('Picture')

    ax2.imshow(mask2)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask2)
    ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score2))
    ax3.set_axis_off()

    for i in range(4):
        n = np.random.choice(len(test_dataset))

        image2, mask2 = test_dataset[n]

        pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)

        print('UNet-EfficientNet-B4 | mIoU {:.3f}'.format(score2))

        visualize(
            image=image2,
            ground_truth=mask2,
            predict_mask = pred_mask2,
        )
