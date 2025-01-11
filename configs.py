# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/9 13:12
# @Software : PyCharm

mnist_cfg = {
    'dataset_type': 'MNIST',
    'img_shape': [1, 28, 28],
    'model_path': './model/ddpm_mnist.pth',
    'batch_size': 512,
    'n_epochs': 50,
    'channels': [10, 20, 40, 80],
    'position_embedding_dim': 128
}

configs = [mnist_cfg]
