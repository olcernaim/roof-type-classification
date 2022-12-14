import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from numpy.random import choice as npc
from torch.utils.data import Dataset


class RoofTrain(Dataset):

    def __init__(self, dataset, transform=None):
        super(RoofTrain, self).__init__()
        np.random.seed(0)
        self.dataset = dataset
        self.transform = transform
        self.img1 = None

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        # file1 = open("Paths.txt", "a")
        # file1.writelines("\n---------------------------------------------\n")
        # file1.close()

        image1 = random.choice(self.dataset.imgs)
        # get image from same class
        label = None
        if index % 2 == 1:
            label = 1.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] == image2[1]:
                    break
        # get image from different class
        else:
            label = 0.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] != image2[1]:
                    break

        # file1.writelines('Index:[%d]\tImage1:[%s]\tImage2:[%s]\tLabel:%d' % (index, image1[0], image2[0], label))
        # file1.close()
        # file1 = open("Paths.txt", "a")
        # file1.writelines('Index:[%d]\tImage1:[%s]\tImage2:[%s]\tLabel:%d' % (index, image1[0], image2[0], label))
        # file1.close()
        path10 = image1[0]
        path20 = image2[0]
        image1 = Image.open(image1[0])
        image2 = Image.open(image2[0])
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        image1 = image1.filter(filter=ImageFilter.GaussianBlur)
        image2 = image2.filter(filter=ImageFilter.GaussianBlur)
        # image1 = image1.filter(filter=ImageFilter.BoxBlur)
        # image2 = image2.filter(filter=ImageFilter.BoxBlur)
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, path10, path20, torch.from_numpy(np.array([label], dtype=np.float32))


class RoofTest(Dataset):

    def __init__(self, dataset, transform=None, times=200, way=20):
        np.random.seed(1)
        super(RoofTest, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.times = times
        self.way = way

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):# times kadar giriyor buraya : Örn :times = 200 ise way = 3 ise , 200 kere 3 resim
        # karşılaştırıyor
        idx = index % self.way
        label = None

        # generate image pair from same class
        if idx == 0:
            self.arr = []
            self.img1 = random.choice(self.dataset.imgs)
            self.arr.append(self.img1[1])
            while True:
                img2 = random.choice(self.dataset.imgs)
                if self.img1[1] == img2[1]:
                    break
        # generate image pair from different class
        else:
            while True:
                img2 = random.choice(self.dataset.imgs)
                if img2[1] not in self.arr:
                    self.arr.append(img2[1])
                    break

        # file1 = open("/content/drive/MyDrive/class.txt", "a")
        # file1.writelines("\n---------------------------------------------\n")
        # file1.writelines('idx:[%d]\tIndex:[%d]\tImage1:[%s]\tImage2:[%s]\tImage1:[%d]\tImage2:[%d]' % (
        # idx, index, self.img1[0], img2[0], self.img1[1], img2[1]))
        # file1.writelines("\n---------------------------------------------\n")
        # file1.close()
        path1 = self.img1[0]
        path2 = img2[0]
        img1 = Image.open(self.img1[0])
        img2 = Image.open(img2[0])
        img1 = img1.convert('L')
        img2 = img2.convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, path1, path2


# test
if __name__ == '__main__':
    roofTrain = RoofTrain('./images_background', 30000 * 8)
    print(roofTrain)
