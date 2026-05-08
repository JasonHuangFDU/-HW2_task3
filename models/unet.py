"""
U-Net 网络结构（从零手写实现，不使用任何预训练权重）

整体结构：
    输入 -> 编码器(4 次下采样, 每级 DoubleConv + MaxPool)
         -> 瓶颈(DoubleConv)
         -> 解码器(4 次上采样, 每级 UpConv + Skip 拼接 + DoubleConv)
         -> 1x1 卷积输出 num_classes 通道

参考: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两次 (Conv3x3 -> BN -> ReLU)，U-Net 编码/解码每一级的基本单元。"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """下采样: MaxPool2d(2) + DoubleConv。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """上采样: 转置卷积或双线性插值 + Skip 拼接 + DoubleConv。"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        x_dec = self.up(x_dec)
        # 由于输入尺寸不一定能被 16 整除, 解码端可能与对应编码端的特征图存在 1 像素差,
        # 这里对 x_dec 做对称 padding 与 x_enc 对齐, 然后沿通道维拼接 (Skip Connection)
        diff_y = x_enc.size(2) - x_dec.size(2)
        diff_x = x_enc.size(3) - x_dec.size(3)
        x_dec = F.pad(
            x_dec,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x_enc, x_dec], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 卷积, 把通道数映射为类别数。"""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """完整的 U-Net 网络。

    Args:
        in_channels: 输入图像通道数, 默认 3 (RGB)
        num_classes: 像素级分类的类别数
        base_channels: 第一层卷积的输出通道数, 后续按 2 倍放大. 论文取 64
        bilinear: 上采样是否使用双线性插值; False 时使用转置卷积
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        base_channels: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        # 双线性插值版瓶颈通道数减半, 用于配平参数量
        factor = 2 if bilinear else 1

        # 编码器
        self.in_conv = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5 // factor)

        # 解码器 (Skip Connection)
        self.up1 = Up(c5, c4 // factor, bilinear)
        self.up2 = Up(c4, c3 // factor, bilinear)
        self.up3 = Up(c3, c2 // factor, bilinear)
        self.up4 = Up(c2, c1, bilinear)

        # 输出
        self.out_conv = OutConv(c1, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        # 全部使用随机初始化, 不加载任何预训练权重 (作业基本要求)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


if __name__ == "__main__":
    # 简单的形状自检
    net = UNet(in_channels=3, num_classes=8)
    dummy = torch.randn(2, 3, 320, 240)
    out = net(dummy)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"输入 shape : {dummy.shape}")
    print(f"输出 shape : {out.shape}")
    print(f"参数量      : {n_params / 1e6:.2f} M")
