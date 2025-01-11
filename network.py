# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/9 13:18
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        """
        位置编码
        :param max_seq_len: 序列长度
        :param d_model: 位置编码的维数
        """
        super(PositionalEncoding, self).__init__()

        # 假设位置编码的维数是偶数
        assert d_model % 2 == 0

        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)

        """
        pos, two_i = torch.meshgrid(i_seq, j_seq):
            i_seq: tensor([0., 1., 2., 3., 4.])
            j_seq: tensor([0., 2., 4., 6.])
            pos: 每一行表示sentence中的一个token; 每一列表示每一个token中的维数
            tensor([[0., 0., 0., 0.],
                    [1., 1., 1., 1.],
                    [2., 2., 2., 2.],
                    [3., 3., 3., 3.],
                    [4., 4., 4., 4.]])
            two_i: sentence中每一个token对应的维数
            tensor([[0., 2., 4., 6.],
                    [0., 2., 4., 6.],
                    [0., 2., 4., 6.],
                    [0., 2., 4., 6.],
                    [0., 2., 4., 6.]])
        """
        pos, two_i = torch.meshgrid(i_seq, j_seq)

        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        """
        torch.stack(): 沿着一个新维度对输入张量序列进行连接，序列中所有的张量都应该为相同形状
        """
        position_encoding = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = position_encoding
        self.embedding.requires_grad_(False)

    def forward(self, x):
        return self.embedding(x)


def create_norm(norm_type, shape):
    """
    归一化方式
    :param norm_type: 归一化类型
    :param shape: (prev_channel, h, w)
    :return:
    """
    if norm_type == 'layer_norm':
        """
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)
        """
        return nn.LayerNorm(shape)
    elif norm_type == 'group_norm':
        """
        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
        """
        return nn.GroupNorm(32, shape[0])


def create_activation(activation_type):
    """
    激活函数
    :param activation_type: 激活函数类型
    :return:
    """
    if activation_type == 'ReLU':
        return nn.ReLU()
    elif activation_type == 'SiLU':
        """
        SiLU: Swish函数
            y = x · \sigma(x)
        """
        return nn.SiLU()


class ResBlock(nn.Module):
    def __init__(self,
                 shape,
                 in_channel,
                 out_channel,
                 time_channel,
                 norm_type='layer_norm',
                 activation_type='SiLU'):
        """
        残差模块
        :param shape: (prev_channel, h, w)
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param time_channel: 隐层通道数
        :param norm_type: 归一化类型
        :param activation_type: 激活函数类型
        """
        super(ResBlock, self).__init__()

        self.norm1 = create_norm(norm_type=norm_type, shape=shape)
        self.norm2 = create_norm(norm_type=norm_type, shape=(out_channel, *shape[1:]))

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

        self.time_layer = nn.Linear(time_channel, out_channel)

        self.activation = create_activation(activation_type)

        if in_channel == out_channel:
            self.residual_conv = nn.Identity()

        else:
            self.residual_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)

    def forward(self, x, time):
        """

        :param x: 输入数据
        :param time: 时间数据 time.shape = [batch_size, 1, d_model]
                     例如在扩散模型中，表示不同的时间步或噪声水平等信息，模型可以根据这个时间信息来调整生成过程
        :return:
        """
        batch_size = time.shape[0]

        # 在将数据输入神经网络之前，先进行归一化处理
        out = self.activation(self.norm1(x))
        out = self.conv1(out)

        time = self.activation(time)
        time = self.time_layer(time).reshape(batch_size, -1, 1, 1)
        out = out + time

        out = self.activation(self.norm2(out))
        out = self.conv2(out)
        out = out + self.residual_conv(x)

        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, shape, dim, norm_type='layer_norm'):
        """
        自注意力机制模块
        :param shape: (channel, height, width)
        :param dim: 维数
        :param norm_type: 归一化类型
        """
        super(SelfAttentionBlock, self).__init__()

        self.norm = create_norm(norm_type=norm_type, shape=shape)

        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

        self.out = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        batch_size, channel, height, width = x.shape

        norm_x = self.norm(x)

        q = self.q(norm_x)
        k = self.k(norm_x)
        v = self.v(norm_x)

        # n c h w -> n h*w c
        q = einops.rearrange(q, "n c h w -> n (h w) c")
        # n c h w -> n c h*w
        k = einops.rearrange(k, "n c h w -> n c (h w)")
        # n c h w -> n c h*w
        v = einops.rearrange(v, "n c h w -> n c (h w)")

        qk = torch.einsum("n i c, n c j -> n j c", q, k) / channel ** 0.5
        qk = torch.softmax(qk, dim=-1)

        result = torch.einsum("n j i, n i c -> n j c", qk, v)
        result = einops.rearrange(result, "n (h w) c -> n c h w", h=height, w=width)

        result = self.out(result)

        return x + result


class ResAttnBlock(nn.Module):
    def __init__(self,
                 shape,
                 in_channel,
                 out_channel,
                 time_channel,
                 with_attn,
                 norm_type='layer_norm',
                 activation_type='SiLU'):
        """
        残差注意力模块
        :param shape: (channel, h, w)
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param time_channel: 隐层维数
        :param with_attn: 注意力机制
        :param norm_type: 归一化类型
        :param activation_type: 激活函数类型
        """
        super(ResAttnBlock, self).__init__()

        self.res_block = ResBlock(shape, in_channel, out_channel, time_channel, norm_type, activation_type)

        if with_attn:
            self.attn_block = SelfAttentionBlock(shape=(out_channel, shape[1], shape[2]), dim=out_channel,
                                                 norm_type=norm_type)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, time):
        """

        :param x: 输入数据
        :param time: 时间数据 time.shape = [batch_size, 1, d_model]
                     例如在扩散模型中，表示不同的时间步或噪声水平等信息，模型可以根据这个时间信息来调整生成过程
        :return:
        """
        x = self.res_block(x, time)
        x = self.attn_block(x)

        return x


class ResAttnBlockMid(nn.Module):
    def __init__(self,
                 shape,
                 in_channel,
                 out_channel,
                 time_channel,
                 with_attn,
                 norm_type='layer_norm',
                 activation_type='SiLU'):
        """

        :param shape: (channel, h, w)
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param time_channel: 隐层维数
        :param with_attn: 注意力机制
        :param norm_type: 归一化类型
        :param activation_type: 激活函数类型
        """
        super().__init__()

        self.res_block1 = ResBlock(shape=shape, in_channel=in_channel, out_channel=out_channel,
                                   time_channel=time_channel, norm_type=norm_type, activation_type=activation_type)

        self.res_block2 = ResBlock(shape=(out_channel, shape[1], shape[2]), in_channel=out_channel,
                                   out_channel=out_channel, time_channel=time_channel, norm_type=norm_type,
                                   activation_type=activation_type)

        if with_attn:
            self.attn_block = SelfAttentionBlock(shape=(out_channel, shape[1], shape[2]), dim=out_channel,
                                                 norm_type=norm_type)

        else:
            self.attn_block = nn.Identity()

    def forward(self, x, time):
        """

        :param x: 输入数据
        :param time: 时间数据 time.shape = [batch_size, 1, d_model]
                     例如在扩散模型中，表示不同的时间步或噪声水平等信息，模型可以根据这个时间信息来调整生成过程
        :return:
        """
        x = self.res_block1(x, time)
        x = self.attn_block(x)
        x = self.res_block2(x, time)
        return x


class UNet(nn.Module):
    def __init__(self,
                 n_steps,
                 img_shape,
                 channels=None,
                 pe_dim=10,
                 with_attention=False,
                 norm_type="layer_norm",
                 activation_type="SiLU"):
        """
        UNet网络
        :param n_steps: 扩散步数
        :param img_shape: image的shape
        :param channels: image的通道数
        :param pe_dim: position embedding dim
        :param with_attention: 注意力机制
        :param norm_type: 归一化类型
        :param activation_type: 激活函数类型
        """
        super(UNet, self).__init__()

        if channels is None:
            channels = [10, 20, 40, 80]

        C, H, W = img_shape

        # 根据通道数量确定神经网络的层数
        layers = len(channels)

        height, width = [H], [W]

        h, w = H, W

        self.NUM_RES_BLOCK = 2

        for _ in range(layers - 1):
            h //= 2
            w //= 2
            height.append(h)
            width.append(w)

        if isinstance(with_attention, bool):
            with_attention = [with_attention] * layers

        self.position_embedding = PositionalEncoding(max_seq_len=n_steps, d_model=pe_dim)

        time_channel = 4 * channels[0]

        self.pe_linear = nn.Sequential(nn.Linear(pe_dim, time_channel),
                                       create_activation(activation_type=activation_type),
                                       nn.Linear(time_channel, time_channel))

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # 下采样
        self.downs = nn.ModuleList()
        # 上采样
        self.ups = nn.ModuleList()

        # 初始通道数量
        prev_channel = channels[0]

        for channel, h, w, with_attn in zip(channels[:-1], height[:-1], width[:-1], with_attention[:-1]):
            encoder_layer = nn.ModuleList()
            for index in range(self.NUM_RES_BLOCK):
                if index != 0:
                    prev_channel = channel
                module = ResAttnBlock((prev_channel, h, w), in_channel=prev_channel, out_channel=channel,
                                      time_channel=time_channel, with_attn=with_attn, norm_type=norm_type,
                                      activation_type=activation_type)

                encoder_layer.append(module)
            self.encoders.append(encoder_layer)
            self.downs.append(nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=2, stride=2))
            prev_channel = channel

        h, w = height[-1], width[-1]
        channel = channels[-1]

        self.mid = ResAttnBlockMid((prev_channel, h, w), in_channel=prev_channel, out_channel=channel,
                                   time_channel=time_channel, with_attn=with_attention[-1], norm_type=norm_type,
                                   activation_type=activation_type)

        prev_channel = channel

        for channel, h, w, with_attn in zip(channels[-2::-1], height[-2::-1], width[-2::-1], with_attention[-2::-1]):
            """
            nn.ConvTranspose2d():
                \mathrm{H_{out}~=~(H_{in}-1)\times stride}[0]-2\times\mathrm{padding}[0]+\mathrm{kernel}\_\mathrm{size}[0]
                \mathrm{W}_{\mathrm{out}}=(\mathrm{W}_{\mathrm{in}}-1)\times\mathrm{stride}[1]-2\times\mathrm{padding}[1]+\mathrm{kernel}\_\mathrm{size}[1]
            """
            self.ups.append(nn.ConvTranspose2d(in_channels=prev_channel, out_channels=channel, kernel_size=2, stride=2))

            decoder_layer = nn.ModuleList()
            for _ in range(self.NUM_RES_BLOCK):
                """
                残差块的输入通道数为·2*channel: 在跳跃连接中会将编码器层的特征图与上采样后的特征图拼接起来
                输出通道数为channel
                """
                module = ResAttnBlock(shape=(2 * channel, h, w), in_channel=2 * channel, out_channel=channel,
                                      time_channel=time_channel, with_attn=with_attention[-1], norm_type=norm_type,
                                      activation_type=activation_type)

                decoder_layer.append(module)

            self.decoders.append(decoder_layer)
            prev_channel = channel

        self.conv_in = nn.Conv2d(in_channels=C, out_channels=channels[0], kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels=prev_channel, out_channels=C, kernel_size=3, stride=1, padding=1)
        self.activation = create_activation(activation_type)

    def forward(self, x, time):
        """

        :param x: 输入数据
        :param time: 时间数据 time.shape = [batch_size, 1]
                     torch.randint(0, 100, (10,)):
                     => tensor([70, 47, 65, 62, 87, 18, 85, 83, 12, 74])
        :return:
        """
        # time.shape = [batch_size, 1, d_model]
        time = self.position_embedding(time)
        # position_embedding.shape = [batch_size, 1, time_channel]
        position_embedding = self.pe_linear(time)

        # x.shape = [batch_size, channels[0], height, width]
        x = self.conv_in(x)

        encoder_outs = []
        for encoder, down in zip(self.encoders, self.downs):
            tmp_outs = []

            for index in range(self.NUM_RES_BLOCK):
                x = encoder[index](x, position_embedding)
                tmp_outs.append(x)

            tmp_outs = list(reversed(tmp_outs))
            encoder_outs.append(tmp_outs)

            # x.shape = [batch_size, channels[i+1], height/2(i+1), width/2(i+1)]
            # 其中，i表示当前循环的次数
            x = down(x)

        x = self.mid(x, position_embedding)
        for decoder, up, encoder_out in zip(self.decoders, self.ups, encoder_outs[::-1]):
            # 上采样，将低分辨率的特征图恢复到较高的分辨率
            x = up(x)

            # 在经过下采样和上采样操作后，上采样后的特征图 x 的尺寸可能与对应的编码器输出 encoder_out 的尺寸不一致
            pad_h = encoder_out[0].shape[2] - x.shape[2]
            pad_w = encoder_out[0].shape[3] - x.shape[3]
            """
            pad_h // 2, pad_h - pad_h // 2: 在高度方向上左侧和右侧需要填充的像素数
            pad_w // 2, pad_w - pad_w // 2: 在宽度方向上左侧和右侧需要填充的像素数
            """
            x = F.pad(x, (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2))

            for index in range(self.NUM_RES_BLOCK):
                resnet_encoder_out = encoder_out[index]
                x = torch.cat((resnet_encoder_out, x), dim=1)
                x = decoder[index](x, position_embedding)

        x = self.conv_out(self.activation(x))
        return x
