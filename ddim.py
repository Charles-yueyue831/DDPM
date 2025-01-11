# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/9 20:18
# @Software : PyCharm

import torch
from tqdm import tqdm

from ddpm import DDPM


class DDIM(DDPM):
    def __init__(self, device, n_steps, min_beta=0.0001, max_beta=0.02):
        """
        DDIM
        :param device: 设备 CPU or GPU
        :param n_steps: 时间步
        :param min_beta: beta的最小值
        :param max_beta: beta的最大值
        """
        """
        DDIM的正向扩散过程和DDPM的正向扩散过程相同
        """
        super(DDIM, self).__init__(device=device, n_steps=n_steps)

    def sample_backward(self, image_or_shape, net, device, simple_var=True, ddim_step=20, eta=1):
        """
        逆向扩散过程
        :param image_or_shape: 图片数据 or shape
        :param net: 网络
        :param device: 设备 CPU or GPU
        :param simple_var: 方差
        :param ddim_step: ddim采样次数
        :param eta:
        :return:
        """
        if simple_var:
            eta = 1

        times = torch.linspace(self.n_steps, 0, ddim_step + 1).to(device).to(torch.long)

        if isinstance(image_or_shape, torch.Tensor):
            x = image_or_shape

        else:
            # 随机高斯噪声
            x = torch.randn(image_or_shape).to(device)

        batch_size = x.shape[0]

        net = net.to(device)

        for i in tqdm(range(1, ddim_step + 1), "DDIM sampling"):
            # 当前时间步x_t
            current_time = times[i - 1] - 1
            # 上一个时间步x_{t-1}
            prev_time = times[i] - 1

            # 时间步x_t的\bar{alpha}
            current_alpha_bar = self.alpha_bars[current_time]
            # 上一个时间步x_{t-1}的\bar{alpha}
            prev_alpha_bar = self.alpha_bars[prev_time] if prev_time >= 0 else 1

            # time_tensor.shape = [batch_size, 1]
            time_tensor = torch.tensor([current_time] * batch_size, dtype=torch.long).to(device).unsqueeze(1)

            epsilon = net(x, time_tensor)

            # DDIM采样过程中的方差
            var = eta * (1 - prev_alpha_bar) / (1 - current_alpha_bar) * (1 - current_alpha_bar / prev_alpha_bar)
            # DDIM采样过程中的噪声
            noise = torch.randn_like(x)

            first_term = (prev_alpha_bar / current_alpha_bar) ** 0.5 * x
            second_term = ((1 - prev_alpha_bar - var) ** 0.5 - (
                    prev_alpha_bar * (1 - current_alpha_bar) / current_alpha_bar) ** 0.5) * epsilon

            if simple_var:
                third_term = (1 - current_alpha_bar / prev_alpha_bar) ** 0.5 * noise
            else:
                third_term = var ** 0.5 * noise

            x = first_term + second_term + third_term

        return x
