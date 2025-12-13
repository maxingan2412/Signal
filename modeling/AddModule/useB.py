"""
GAM (Global Alignment Module) + LAM (Local Alignment Module) - 对齐模块

实现论文中的:
1. GAM: 全局对齐模块 - 通过 3D 多面体体积最小化实现跨模态全局特征对齐
2. LAM: 局部对齐模块 - 通过可变形空间采样 (DAS) 实现像素级对齐

对应论文公式 (19)-(26)
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.volume import volume_computation3  # 计算 3D 多面体体积
from modeling.AddModule.DAS import DA_sample as DAS  # 可变形空间采样
import math


class QuickGELU(nn.Module):
    """
    快速 GELU 激活函数
    使用 sigmoid 近似，比标准 GELU 更快
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AlignmentM(nn.Module):
    """
    对齐模块 (AlignmentM)

    包含两个子模块:
    1. GAM (Cls_Align): 全局对齐 - 最小化三模态特征构成的 3D 多面体体积
    2. LAM (patch_Align): 局部对齐 - 可变形采样后计算 MSE 损失

    输入:
        RGB_patch, NI_patch, TI_patch: [B, 128, 512] 三模态 patch tokens

    输出:
        stage="CLS": 只返回 GAM loss (标量)
        stage="together_CLS_Patch": 返回 (GAM loss, LAM loss)
    """

    def __init__(self, feat_dim, H, W):
        """
        参数:
            feat_dim: 特征维度 (512)
            H: patch 网格高度 (16, 来自 256/16)
            W: patch 网格宽度 (8, 来自 128/16)
        """
        super(AlignmentM, self).__init__()
        self.feat_dim = feat_dim

        # ============ GAM 参数 ============
        # 对比学习温度参数 (可学习)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))

        # ============ LAM 参数 ============
        self.h, self.w = H, W  # 16, 8
        self.mse = nn.MSELoss()  # MSE 损失用于局部对齐

        # DAS (Deformable Attention Sampling) 参数
        n_heads = 1           # 注意力头数
        n_head_channels = 512  # 每个头的通道数
        n_groups = 1          # 分组数
        stride = 4            # 下采样步长
        offset_range_factor = 2  # 偏移范围因子
        ksize = 4             # 卷积核大小

        # 三个模态各自的可变形采样模块
        # 每个模块学习自己模态的空间偏移量
        self.DAS_r = DAS(n_heads, n_head_channels, n_groups, stride, offset_range_factor, ksize)
        self.DAS_n = DAS(n_heads, n_head_channels, n_groups, stride, offset_range_factor, ksize)
        self.DAS_t = DAS(n_heads, n_head_channels, n_groups, stride, offset_range_factor, ksize)

    def Cls_Align(self, RGB_patch, NI_patch, TI_patch):
        """
        GAM: 全局对齐模块 - 对应论文公式 (19)-(22)

        原理: 计算三模态特征在 Gramian 空间中构成的 3D 多面体体积
        体积越小说明三个模态越对齐

        输入:
            RGB_patch, NI_patch, TI_patch: [B, 128, 512] patch tokens

        输出:
            loss_area: 标量, GAM 损失
        """
        # ============ 第1步: 计算全局特征 (平均池化) ============
        # 对所有 patch tokens 取平均得到全局特征
        # [B, 128, 512] -> [B, 512]
        rgb_feature = torch.mean(RGB_patch, dim=1)  # [B, 512]
        ni_feature = torch.mean(NI_patch, dim=1)    # [B, 512]
        ti_feature = torch.mean(TI_patch, dim=1)    # [B, 512]

        # ============ 第2步: L2 归一化 ============
        # 将特征投影到单位超球面上
        feat_ro = F.normalize(rgb_feature, dim=-1)  # [B, 512], ||feat|| = 1
        feat_no = F.normalize(ni_feature, dim=-1)
        feat_to = F.normalize(ti_feature, dim=-1)

        # ============ 第3步: 计算 3D 多面体体积矩阵 ============
        # volume_computation3 计算三个向量构成的平行六面体体积
        # 公式 (20): V = det([f_R, f_N, f_T])
        # V: [B, B] 矩阵，V[i,j] 表示第 i 个样本的 RGB 与第 j 个样本的 NI/TI 的体积
        V = volume_computation3(feat_ro, feat_no, feat_to)
        volume = V / self.contra_temp  # 温度缩放

        # 转置版本用于对称损失
        VT = volume_computation3(feat_ro, feat_no, feat_to).T
        volumeT = VT / self.contra_temp

        # ============ 第4步: 计算对比损失 ============
        # 目标: 对角线元素（同一样本的三模态）应该最小
        # 使用交叉熵损失，标签为对角线索引
        bs = feat_ro.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(feat_ro.device)  # [0, 1, 2, ..., B-1]

        # 公式 (22): L_GAM = CE(-V, targets) + CE(-V^T, targets)
        # 负号是因为我们要最小化体积（体积小 = 对齐好）
        loss_area = (
            F.cross_entropy(-volume, targets, label_smoothing=0.1)   # d2a: 沿第一个维度
            + F.cross_entropy(-volumeT, targets, label_smoothing=0.1)  # a2d: 沿转置维度
        ) / 2

        return loss_area

    def patch_Align(self, RGB_patch, NI_patch, TI_patch):
        """
        LAM: 局部对齐模块 - 对应论文公式 (23)-(26)

        原理: 使用可变形空间采样 (DAS) 学习像素级偏移量
        采样后的特征应该对齐，通过 MSE 损失约束

        输入:
            RGB_patch, NI_patch, TI_patch: [B, 128, 512] patch tokens

        输出:
            pat_loss: 标量, LAM 损失
        """
        bs, n, dim = RGB_patch.shape  # B, 128, 512

        # ============ 第1步: reshape 为 2D 特征图 ============
        # [B, 128, 512] -> [B, 16, 8, 512] -> [B, 512, 16, 8]
        # 将 patch tokens 重排为空间网格形式
        RGB_patch = RGB_patch.reshape(bs, self.h, self.w, -1).permute(0, 3, 1, 2)  # [B, 512, 16, 8]
        NI_patch = NI_patch.reshape(bs, self.h, self.w, -1).permute(0, 3, 1, 2)
        TI_patch = TI_patch.reshape(bs, self.h, self.w, -1).permute(0, 3, 1, 2)

        # ============ 第2步: 可变形空间采样 (DAS) ============
        # 每个模态的 DAS 模块学习自己的空间偏移量
        # 然后在偏移后的位置进行双线性采样
        # 公式 (24): x_sampled = grid_sample(x, pos + offset)
        rgb_sampled = self.DAS_r(RGB_patch)  # [B, 512, H', W'] H'=4, W'=2 (stride=4)
        nir_sampled = self.DAS_n(NI_patch)
        tir_sampled = self.DAS_t(TI_patch)

        # ============ 第3步: 计算成对 MSE 损失 ============
        # 公式 (26): L_LAM = MSE(f_R, f_N) + MSE(f_R, f_T) + MSE(f_N, f_T)
        # 采样后的特征应该对齐（同一位置表示同一语义区域）
        loss1 = self.mse(nir_sampled, rgb_sampled)  # NIR vs RGB
        loss2 = self.mse(tir_sampled, rgb_sampled)  # TIR vs RGB
        loss3 = self.mse(tir_sampled, nir_sampled)  # TIR vs NIR

        pat_loss = (loss1 + loss2 + loss3) / 3

        return pat_loss

    def forward(self, RGB_patch, NI_patch, TI_patch, stage):
        """
        对齐模块前向传播

        参数:
            RGB_patch, NI_patch, TI_patch: [B, 128, 512] 三模态 patch tokens
            stage: "CLS" 只计算 GAM, "together_CLS_Patch" 计算 GAM + LAM

        返回:
            stage="CLS": GAM loss (标量)
            stage="together_CLS_Patch": (GAM loss, LAM loss)
        """
        if stage == "CLS":
            # 只计算全局对齐损失
            cls_loss = self.Cls_Align(RGB_patch, NI_patch, TI_patch)
            return cls_loss

        # 计算全局 + 局部对齐损失
        loss_area = self.Cls_Align(RGB_patch, NI_patch, TI_patch)    # GAM loss
        patch_loss = self.patch_Align(RGB_patch, NI_patch, TI_patch)  # LAM loss

        return loss_area, patch_loss
