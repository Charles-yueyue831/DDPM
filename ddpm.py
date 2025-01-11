# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/9 19:36
# @Software : PyCharm

import torch
from tqdm import tqdm


class DDPM(object):
    def __init__(self, device, n_steps, min_beta=0.0001, max_beta=0.02):
        """
        DDPM
        :param device: 设备 CPU or GPU
        :param n_steps: 时间步
        :param min_beta: beta的最小值
        :param max_beta: beta的最大值
        """
        # 扩散率
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)

        alphas = 1 - betas

        """
        torch.empty_like(alphas): 输出的形状和alpha_bars相同
        """
        alpha_bars = torch.empty_like(alphas)

        product = 1

        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product

        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def sample_forward(self, x, time, epsilon=None):
        """
        正向扩散过程
        :param x: 输入数据
        :param time: 时间数据，表示当前时间步或噪声水平
        :param epsilon: 噪声
        :return:
        """
        # alpha_bar.shape = [batch_size, 1, 1, 1]
        alpha_bar = self.alpha_bars[time].reshape(-1, 1, 1, 1)

        if epsilon is None:
            epsilon = torch.randn_like(x)

        result = epsilon * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return result

    def sample_backward(self, image_or_shape, net, device, simple_var=True):
        """
        逆向扩散过程
        :param image_or_shape: 图片数据 or shape
        :param net: 网络
        :param device: 设备 CPU or GPU
        :param simple_var: 方差
        :return:
        """
        if isinstance(image_or_shape, torch.Tensor):
            x = image_or_shape
        else:
            x = torch.randn(image_or_shape).to(device)

        net = net.to(device)

        for t in tqdm(range(self.n_steps - 1, -1, -1), "DDPM sampling"):
            x = self.sample_backward_step(x_t=x, t=t, net=net, device=device, simple_var=simple_var)

        return x

    def sample_backward_step(self, x_t, t, net, device, simple_var=True):
        """

        :param x_t: 第t个时间步的数据
        :param t: 时间步
        :param net: 网络
        :param device: 设备 CPU or GPU
        :param simple_var: 方差
        :return:
        """
        n = x_t.shape[0]

        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(device).unsqueeze(1)

        # 使用网络预测噪声
        epsilon = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]

            else:
                """
                \sigma^{2}=\frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha_{t}}}
                """
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]

            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * epsilon) / torch.sqrt(self.alphas[t])

        # 重参数技巧
        x_t = mean + noise

        return x_t
