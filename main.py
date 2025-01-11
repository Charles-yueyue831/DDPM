# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/9 13:10
# @Software : PyCharm

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import einops
import cv2
from configs import configs
from network import UNet
from ddpm import DDPM
from ddim import DDIM
from dataset import get_dataloader


def train(model: DDPM, net: UNet, dataset_type, resolution=None, batch_size=512, n_epochs=50, device="cuda",
          ckpt_path="./model/ddpm.pth"):
    """
    训练阶段
    :param model: DDPM
    :param net: UNet模型
    :param dataset_type: 数据集类型
    :param resolution: 分辨率
    :param batch_size: batch size
    :param n_epochs: 训练轮次
    :param device: 设备 CPU or GPU
    :param ckpt_path: 模型保存路径
    :return:
    """
    steps = model.n_steps

    dataloader = get_dataloader(dataset_type=dataset_type, batch_size=batch_size)

    net = net.to(device)

    loss_function = nn.MSELoss()

    optimizer = optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    start_time = time.time()

    for epoch in range(n_epochs):
        total_loss = 0.

        for x in dataloader:
            current_batch_size = x.shape[0]

            x = x.to(device)

            t = torch.randint(0, steps, (current_batch_size,)).to(device)

            # 正向扩散过程中的噪声
            epsilon = torch.randn_like(x).to(device)

            x_t = model.sample_forward(x, time=t, epsilon=epsilon)

            epsilon_theta = net(x_t, t.reshape(current_batch_size, 1))

            # 扩散模型的损失：添加的噪声和预测的噪声之间的损失
            loss = loss_function(epsilon_theta, epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * current_batch_size

        total_loss /= len(dataloader.dataset)

        end_time = time.time()

        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {epoch} loss: {total_loss} elapsed {(end_time - start_time):.2f}s')


def sample_image(model, net, output_path, image_shape, n_sample=64, device="cuda", simple_var=True, to_bgr=False,
                 **kwargs):
    """
    DDPM or DDIM采样阶段
    :param model: DDPM or DDIM
    :param net: UNet
    :param output_path: 采样结果的输出路径
    :param image_shape: 图片形状
    :param n_sample: 样本数量
    :param device: 设备 CPU or GPU
    :param simple_var: 方差
    :param to_bgr: 红绿蓝
    :param kwargs:
    :return:
    """
    if image_shape[1] >= 256:
        max_batch_size = 16
    elif image_shape[1] >= 128:
        max_batch_size = 64
    else:
        max_batch_size = 256

    net = net.to(device)
    net = net.eval()

    index = 0
    with torch.no_grad():
        while n_sample > 0:
            if n_sample >= max_batch_size:
                batch_size = max_batch_size
            else:
                batch_size = n_sample

            n_sample -= batch_size

            shape = (batch_size, *image_shape)

            images = model.sample_backward(shape, net, device=device, simple_var=simple_var, **kwargs).detach().cpu()

            images = (images + 1) / 2 * 255
            images = images.clamp(0, 255).to(torch.uint8)

            images_list = einops.rearrange(images, "n c h w -> n h w c").numpy()

            """
            os.path.splitext(): 将文件路径拆分成文件名和文件扩展名
            """
            output_dir = os.path.splitext(output_path)[0]

            os.makedirs(output_dir, exist_ok=True)

            for i, image in enumerate(images_list):
                if to_bgr:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rf"{output_dir}/{i + index}.jpg", image)

            index += batch_size


if __name__ == "__main__":
    os.makedirs("work_dirs", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # 0 for MNIST. See configs.py
    config_id = 0
    cfg = configs[config_id]

    # 扩散步数
    n_steps = 1000

    gpu_cpu = 'cuda'

    model_path = cfg['model_path']
    img_shape = cfg['img_shape']

    # bgr: 蓝色、绿色和红色
    bgr = False if cfg['dataset_type'] == 'MNIST' else True

    unet = UNet(n_steps=n_steps, img_shape=img_shape, channels=cfg["channels"], pe_dim=cfg["position_embedding_dim"],
                with_attention=cfg.get("with_attention", False), norm_type=cfg.get("norm_type", "layer_norm"))

    ddpm = DDPM(n_steps=n_steps, device=gpu_cpu)

    # train(model=ddpm, net=unet, dataset_type=cfg["dataset_type"], batch_size=cfg["batch_size"],
    #       n_epochs=cfg["n_epochs"], device=gpu_cpu, ckpt_path=model_path)

    unet.load_state_dict(torch.load(model_path))

    ddim = DDIM(device=gpu_cpu, n_steps=n_steps)

    # sample_image(model=ddpm, net=unet, output_path="./work_dirs/ddpm.jpg", image_shape=img_shape, device=gpu_cpu,
    #              to_bgr=bgr)
    sample_image(model=ddim, net=unet, output_path="./work_dirs/ddim.jpg", image_shape=img_shape, device=gpu_cpu,
                 simple_var=False, eta=0, ddim_step=10)
