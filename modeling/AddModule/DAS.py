"""
DAS (Deformable Attention Sampling) - 可变形空间采样模块

实现论文中 LAM 的核心组件:
通过学习空间偏移量，在偏移后的位置进行双线性采样
用于校正多模态图像之间的像素级错位

对应论文公式 (23)-(25)
"""

import torch.nn as nn
import einops
import torch.nn.functional as F
import torch


class DA_sample(nn.Module):
    """
    可变形空间采样 (Deformable Attention Sampling)

    功能:
    1. 学习输入特征的空间偏移量 (offset)
    2. 在偏移后的位置进行双线性采样
    3. 返回采样后的特征图

    输入: [B, C, H, W] 2D 特征图 (H=16, W=8, C=512)
    输出: [B, C, H', W'] 采样后的特征图 (H'=4, W'=2, stride=4)
    """

    def __init__(
            self, n_heads, n_head_channels, n_groups,
            stride, offset_range_factor, ksize
    ):
        """
        参数:
            n_heads: 注意力头数 (1)
            n_head_channels: 每个头的通道数 (512)
            n_groups: 分组数 (1)
            stride: 下采样步长 (4)
            offset_range_factor: 偏移范围因子 (2)
            ksize: 卷积核大小 (4)
        """
        super().__init__()
        self.n_head_channels = n_head_channels  # 512
        self.nc = n_head_channels * n_heads     # 总通道数: 512 * 1 = 512
        self.n_groups = n_groups                # 分组数: 1
        self.n_group_channels = self.nc // self.n_groups  # 每组通道数: 512
        self.kk = ksize                         # 卷积核大小: 4
        self.stride = stride                    # 步长: 4
        self.offset_range_factor = offset_range_factor  # 偏移范围: 2
        pad_size = 0

        # ============ 偏移量预测网络 ============
        # 输入: [B, 512, H, W]
        # 输出: [B, 1, H/stride, W/stride] 偏移量 (只预测一个方向，另一个由参考点提供)
        self.conv_offset = nn.Sequential(
            # 1x1 卷积: 通道间信息交互
            nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),  # [512, 512, 1, 1]
            nn.GELU(),
            # 深度可分离卷积: 空间下采样 + 局部特征提取
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, stride, pad_size,
                      groups=self.n_group_channels),  # [512, 512, 4, 4] groups=512
            nn.GELU(),
            # 1x1 卷积: 输出偏移量 (单通道)
            nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)  # [512, 1, 1, 1]
        )

        # Query 投影
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        """
        生成参考点网格 (Reference Points)

        在采样位置生成均匀分布的参考点
        偏移量将加到这些参考点上得到最终采样位置

        参数:
            H_key, W_key: 输出特征图的高宽 (4, 2)
            B: batch size
            dtype, device: 数据类型和设备

        返回:
            ref: [B*n_groups, H_key, W_key, 2] 参考点坐标 (归一化到 [-1, 1])
        """
        # 创建网格点 (0.5, 1.5, 2.5, ...) 居中采样
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)  # [H_key, W_key, 2]

        # 归一化到 [-1, 1] 范围 (grid_sample 要求)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)  # x: [0, W-1] -> [-1, 1]
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)  # y: [0, H-1] -> [-1, 1]

        # 扩展到 batch 维度
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # [B*g, H, W, 2]

        return ref

    def forward(self, x):
        """
        可变形采样前向传播

        数据流:
        x [B, 512, 16, 8]
          -> proj_q: Query 投影
          -> conv_offset: 预测偏移量 [B, 1, 4, 2]
          -> 偏移量 + 参考点 = 采样位置
          -> grid_sample: 双线性采样
          -> x_sampled [B, 512, 4, 2]

        参数:
            x: [B, C, H, W] 输入特征图 (B, 512, 16, 8)

        返回:
            x_sampled: [B, C, H', W'] 采样后的特征图 (B, 512, 4, 2)
        """
        B, C, H, W = x.size()  # B, 512, 16, 8
        dtype, device = x.dtype, x.device

        # ============ 第1步: Query 投影 ============
        q = self.proj_q(x)  # [B, 512, 16, 8]

        # ============ 第2步: 重排为分组形式 ============
        # [B, 512, 16, 8] -> [B*1, 512, 16, 8] (n_groups=1 时无变化)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)

        # ============ 第3步: 预测偏移量 ============
        offset = self.conv_offset(q_off).contiguous()  # [B*g, 1, H/stride, W/stride] = [B, 1, 4, 2]
        Hk, Wk = offset.size(2), offset.size(3)  # 4, 2

        # ============ 第4步: 限制偏移范围 ============
        if self.offset_range_factor > 0:
            # 将偏移量限制在合理范围内
            # tanh 将输出限制在 [-1, 1]，然后乘以范围因子
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            # 公式: offset = tanh(offset) * (1/(H-1), 1/(W-1)) * factor

        # 重排偏移量: [B, 1, H, W] -> [B, H, W, 1] (需要扩展为2D偏移)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')  # [B, 4, 2, 1]

        # ============ 第5步: 计算采样位置 ============
        # 参考点 + 偏移量 = 最终采样位置
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)  # [B, 4, 2, 2]
        pos = (offset + reference).clamp(-1., +1.)  # [B, 4, 2, 2] 限制在 [-1, 1] 范围

        # ============ 第6步: 双线性采样 ============
        # grid_sample 在 pos 指定的位置从 x 中采样
        # 公式 (25): x_sampled = grid_sample(x, pos)
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),  # [B, 512, 16, 8]
            grid=pos[..., (1, 0)],  # [B, 4, 2, 2] 注意: grid_sample 要求 (x, y) 顺序
            mode='bilinear',
            align_corners=True
        )  # [B, 512, 4, 2]

        return x_sampled
