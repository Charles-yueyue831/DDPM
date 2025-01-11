# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/10 22:17
# @Software : PyCharm

import os
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def download_dataset():
    """
    下载数据集
    :return:
    """
    mnist = datasets.MNIST(root=rf"E:\python\dataset\mnist", download=False)


class ImageDataset(Dataset):
    def __init__(self, dataset_type="MNIST"):
        """
        图像数据集
        :param dataset_type: 数据集类型
        """
        super(ImageDataset, self).__init__()

        self.dataset_type = dataset_type

        self.dataset = None

        if self.dataset_type == "MNIST":
            self.dataset = datasets.MNIST(root=rf"E:\python\dataset\mnist")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        获取数据集中指定索引的数据
        :param index: 索引
        :return:
        """

        if self.dataset_type == "MNIST":
            image = self.dataset[index][0]

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - 0.5) * 2)
            ])

            return transform(image)

        return None


def get_dataloader(dataset_type, batch_size, num_workers=4):
    """
    获取dataloader
    :param dataset_type: 数据集类型
    :param batch_size: batch size
    :param num_workers: 线程数量
    :return:
    """
    dataset = ImageDataset(dataset_type=dataset_type)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader
