"""
SIM (Selective Interaction Module) - 选择性交互模块

实现论文中的 Token Selection 和 Modal Interaction 功能:
1. TokenSelection: 模态内/模态间 token 选择，过滤背景噪声
2. ModalInteractive: 跨模态特征融合

对应论文公式 (5)-(18)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenSelection(nn.Module):
    """
    Token 选择模块

    功能:
    - intra_modal_token_selection: 模态内选择，CLS token 关注自己模态的 patches
    - inter_modal_token_selection: 模态间选择，CLS token 关注其他模态的 patches

    输入:
        rgb_patches, nir_patches, tir_patches: [B, L, dim] L=128 个 patch tokens
        rgb_global, nir_global, tir_global: [B, dim] CLS tokens

    输出:
        selected_patches: 经过 mask 筛选的 patches [B, L, dim]
    """

    def __init__(self, dim, k=112):
        """
        参数:
            dim: 特征维度 (512 for CLIP ViT-B-16)
            k: 模态内选择的 token 数量 (k1=k, k2=2k 用于模态间)
        """
        super().__init__()
        self.dim = dim
        self.k1 = k      # 模态内选择数量
        self.k2 = 2*k    # 模态间选择数量
        # 用于模态间选择的投影层
        self.W_q = nn.Linear(dim, dim)  # Query 投影
        self.W_k = nn.Linear(dim, dim)  # Key 投影
        self.W_v = nn.Linear(dim, dim)  # Value 投影 (实际未使用)

    def intra_modal_token_selection(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
        """
        模态内 Token 选择 - 对应论文公式 (5)-(7)

        原理: 每个模态的 CLS token 作为 Query，关注自己模态的 patch tokens
        选择注意力分数最高的 k1 个 patches

        输入:
            rgb_patches: [B, L, dim] RGB 模态的 patch tokens
            rgb_global: [B, dim] RGB 模态的 CLS token
            (nir, tir 同理)

        输出:
            rgb_mask, nir_mask, tir_mask: [B, L, 1] 二值 mask
        """
        batch_size = rgb_patches.size(0)

        # ============ 计算注意力分数 ============
        # 公式: score = softmax(CLS · patches^T / √dim)
        # rgb_global.unsqueeze(1): [B, dim] -> [B, 1, dim]
        # rgb_patches.transpose(1, 2): [B, L, dim] -> [B, dim, L]
        # bmm 结果: [B, 1, L] -> squeeze -> [B, L]
        rgb_scores = F.softmax((torch.bmm(rgb_global.unsqueeze(1), rgb_patches.transpose(1, 2))/math.sqrt(self.dim)).squeeze(1), dim=1)
        nir_scores = F.softmax((torch.bmm(nir_global.unsqueeze(1), nir_patches.transpose(1, 2))/math.sqrt(self.dim)).squeeze(1), dim=1)
        tir_scores = F.softmax((torch.bmm(tir_global.unsqueeze(1), tir_patches.transpose(1, 2))/math.sqrt(self.dim)).squeeze(1), dim=1)
        # *_scores: [B, L] 每个 patch 的重要性分数

        # ============ TopK 选择 ============
        # 选择分数最高的 k1 个 patches
        rgb_topk, rgb_indices = torch.topk(rgb_scores, min(self.k1, rgb_scores.size(1)), dim=1)
        nir_topk, nir_indices = torch.topk(nir_scores, min(self.k1, nir_scores.size(1)), dim=1)
        tir_topk, tir_indices = torch.topk(tir_scores, min(self.k1, tir_scores.size(1)), dim=1)
        # *_indices: [B, k1] 被选中的 patch 索引

        # ============ 构建二值 Mask ============
        rgb_mask = torch.zeros_like(rgb_patches[:, :, 0])  # [B, L]
        nir_mask = torch.zeros_like(nir_patches[:, :, 0])
        tir_mask = torch.zeros_like(tir_patches[:, :, 0])

        # 将选中的位置标记为 1
        for i in range(batch_size):
            rgb_mask[i, rgb_indices[i]] = 1
            nir_mask[i, nir_indices[i]] = 1
            tir_mask[i, tir_indices[i]] = 1

        # [B, L] -> [B, L, 1] 便于后续广播乘法
        return rgb_mask.unsqueeze(-1), nir_mask.unsqueeze(-1), tir_mask.unsqueeze(-1)

    def inter_modal_token_selection(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
        """
        模态间 Token 选择 - 对应论文公式 (8)-(13)

        原理: 每个模态的 CLS token 作为 Query，关注【其他模态】的 patch tokens
        选择跨模态注意力分数最高的 k2 个 patches

        关键思想: RGB 的 CLS 选中的 NIR/TIR patches 说明这些区域跨模态一致性高

        输入/输出: 同 intra_modal_token_selection
        """
        batch_size = rgb_patches.size(0)
        n_rgb = rgb_patches.size(1)  # L = 128
        n_nir = nir_patches.size(1)  # L = 128

        # ============ 构建 Query 和 Key ============
        # Query: 三个模态的 CLS tokens 堆叠
        # 公式 (8): Q = T[f_R^c, f_N^c, f_T^c]
        queries = torch.stack([rgb_global, nir_global, tir_global], dim=1)  # [B, 3, dim]

        # Key: 三个模态的 patches 拼接
        # 公式 (9): K = C[f_R^p, f_N^p, f_T^p]
        keys = torch.cat([rgb_patches, nir_patches, tir_patches], dim=1)  # [B, 3L, dim]

        # ============ 线性投影 ============
        q = self.W_q(queries)  # [B, 3, dim]
        k = self.W_k(keys)     # [B, 3L, dim]

        # ============ 计算跨模态注意力分数 ============
        # 公式 (10): S = softmax(QK^T / √dim)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)  # [B, 3, 3L]
        scores = F.softmax(scores, dim=2)
        # scores[b, i, j] 表示第 i 个模态的 CLS 对第 j 个 patch 的注意力

        # ============ 提取跨模态分数（排除自身模态）============
        # 公式 (11): D_m = C̄[S[u ≠ m]] - 只看其他模态的分数

        # RGB 的 CLS 看 NIR 和 TIR 的 patches（排除 RGB 自己的 patches [0:n_rgb]）
        rgb_scores = torch.cat([
            scores[:, 0, n_rgb:n_rgb+n_nir],    # RGB->NIR: [B, n_nir]
            scores[:, 0, n_rgb+n_nir:]          # RGB->TIR: [B, n_tir]
        ], dim=1)  # [B, n_nir + n_tir]

        # NIR 的 CLS 看 RGB 和 TIR 的 patches（排除 NIR 自己的 patches [n_rgb:n_rgb+n_nir]）
        nir_scores = torch.cat([
            scores[:, 1, :n_rgb],               # NIR->RGB: [B, n_rgb]
            scores[:, 1, n_rgb+n_nir:]          # NIR->TIR: [B, n_tir]
        ], dim=1)  # [B, n_rgb + n_tir]

        # TIR 的 CLS 看 RGB 和 NIR 的 patches（排除 TIR 自己的 patches [n_rgb+n_nir:]）
        tir_scores = torch.cat([
            scores[:, 2, :n_rgb],               # TIR->RGB: [B, n_rgb]
            scores[:, 2, n_rgb:n_rgb+n_nir]     # TIR->NIR: [B, n_nir]
        ], dim=1)  # [B, n_rgb + n_nir]

        # ============ TopK 选择 ============
        # 公式 (12): Θ̄_m = TopK(D_m, k2)
        rgb_topk, rgb_indices = torch.topk(rgb_scores, min(self.k2, rgb_scores.size(1)), dim=1)
        nir_topk, nir_indices = torch.topk(nir_scores, min(self.k2, nir_scores.size(1)), dim=1)
        tir_topk, tir_indices = torch.topk(tir_scores, min(self.k2, tir_scores.size(1)), dim=1)
        # rgb_indices: RGB 选中的 NIR/TIR patches 在拼接后的索引

        # ============ 构建二值 Mask ============
        rgb_mask = torch.zeros_like(rgb_patches[:, :, 0])  # [B, L]
        nir_mask = torch.zeros_like(nir_patches[:, :, 0])
        tir_mask = torch.zeros_like(tir_patches[:, :, 0])

        # 遍历batch中的每个样本，将跨模态选择结果"反向映射"到被选中模态的mask上
        for i in range(batch_size):

            # ============ 第一部分：处理 RGB 的 CLS token 选中的 patches ============
            # rgb_indices 来自 rgb_scores = [NIR patches分数, TIR patches分数] 的 TopK
            # 索引值域: [0, n_nir) 对应NIR, [n_nir, n_nir+n_tir) 对应TIR

            # 判断哪些索引属于NIR范围 (索引 < n_nir 说明选中的是NIR的patch)
            nir_selected = rgb_indices[i] < n_nir
            # 剩余的属于TIR范围
            tir_selected = ~nir_selected
            # 提取NIR中被选中的位置 (索引值本身就是NIR内的位置)
            nir_pos = rgb_indices[i][nir_selected]
            # 提取TIR中被选中的位置 (需减去n_nir偏移量，还原为TIR内的真实位置)
            tir_pos = rgb_indices[i][tir_selected] - n_nir
            # 在NIR/TIR的mask上标记: "RGB认为这些patches重要"
            if nir_pos.numel() > 0:
                nir_mask[i, nir_pos] = 1
            if tir_pos.numel() > 0:
                tir_mask[i, tir_pos] = 1

            # ============ 第二部分：处理 NIR 的 CLS token 选中的 patches ============
            # nir_indices 来自 nir_scores = [RGB patches分数, TIR patches分数] 的 TopK
            # 索引值域: [0, n_rgb) 对应RGB, [n_rgb, n_rgb+n_tir) 对应TIR

            # 判断哪些索引属于RGB范围
            nir_selected = nir_indices[i] < n_rgb  # 变量复用，实际含义是rgb_selected
            tir_selected = ~nir_selected
            # 提取RGB中被选中的位置
            rgb_pos = nir_indices[i][nir_selected]
            # 提取TIR中被选中的位置 (减去n_rgb偏移量)
            tir_pos = nir_indices[i][tir_selected] - n_rgb
            # 在RGB/TIR的mask上标记: "NIR认为这些patches重要"
            if rgb_pos.numel() > 0:
                rgb_mask[i, rgb_pos] = 1
            if tir_pos.numel() > 0:
                tir_mask[i, tir_pos] = 1

            # ============ 第三部分：处理 TIR 的 CLS token 选中的 patches ============
            # tir_indices 来自 tir_scores = [RGB patches分数, NIR patches分数] 的 TopK
            # 索引值域: [0, n_rgb) 对应RGB, [n_rgb, n_rgb+n_nir) 对应NIR

            # 判断哪些索引属于RGB范围
            rgb_selected = tir_indices[i] < n_rgb
            nir_selected = ~rgb_selected
            # 提取RGB中被选中的位置
            rgb_pos = tir_indices[i][rgb_selected]
            # 提取NIR中被选中的位置 (减去n_rgb偏移量)
            nir_pos = tir_indices[i][nir_selected] - n_rgb
            # 在RGB/NIR的mask上标记: "TIR认为这些patches重要"
            if rgb_pos.numel() > 0:
                rgb_mask[i, rgb_pos] = 1
            if nir_pos.numel() > 0:
                nir_mask[i, nir_pos] = 1

        # 添加最后一维 [B, L] -> [B, L, 1]，便于后续与patches特征相乘
        return rgb_mask.unsqueeze(-1), nir_mask.unsqueeze(-1), tir_mask.unsqueeze(-1)

    def forward(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
        """
        Token 选择前向传播

        输入:
            rgb_patches, nir_patches, tir_patches: [B, 128, 512] patch tokens
            rgb_global, nir_global, tir_global: [B, 512] CLS tokens

        输出:
            rgb_selected, nir_selected, tir_selected: [B, 128, 512] 筛选后的 patches
            (未选中的位置被置为 0)
        """
        # 1. 模态间选择: 其他模态认为重要的 patches
        rgb_mask_c, nir_mask_c, tir_mask_c = self.inter_modal_token_selection(
            rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global
        )
        # *_mask_c: [B, L, 1] 跨模态选择的 mask

        # 2. 模态内选择: 自己模态认为重要的 patches
        rgb_mask_i, nir_mask_i, tir_mask_i = self.intra_modal_token_selection(
            rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global
        )
        # *_mask_i: [B, L, 1] 模态内选择的 mask

        # 3. Union 合并: 只要任一方式选中即保留
        # 公式 (14): M_m = M_m^c ∪ M_m^s
        rgb_mask = (rgb_mask_c + rgb_mask_i > 0).float()  # [B, L, 1]
        nir_mask = (nir_mask_c + nir_mask_i > 0).float()
        tir_mask = (tir_mask_c + tir_mask_i > 0).float()

        # 4. 应用 mask 筛选 patches
        # 公式 (15): f̃_m^p = f_m^p ⊙ M_m
        rgb_selected = rgb_patches * rgb_mask  # [B, 128, 512] 未选中位置为 0
        nir_selected = nir_patches * nir_mask
        tir_selected = tir_patches * tir_mask

        return rgb_selected, nir_selected, tir_selected


class ModalInteractive(nn.Module):
    """
    模态交互模块 - 对应论文公式 (16)-(18)

    功能: 使用交叉注意力将选中的 patches 信息融合到 CLS tokens 中

    结构:
    - Multi-Head Cross Attention: CLS tokens 作为 Query，patches 作为 Key/Value
    - FFN: 前馈神经网络
    - 残差连接 + LayerNorm
    """

    def __init__(self, dim, num_heads=8):
        """
        参数:
            dim: 特征维度 (512)
            num_heads: 注意力头数 (8)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # 前馈神经网络 FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2*dim),   # [dim] -> [2*dim]
            nn.GELU(),
            nn.Linear(2*dim, dim)    # [2*dim] -> [dim]
        )

        # LayerNorm
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, rgb_selected, nir_selected, tir_selected, rgb_global, nir_global, tir_global):
        """
        模态交互模块的前向传播

        输入:
            rgb_selected, nir_selected, tir_selected: [B, L, dim] 经过 TokenSelection 筛选后的 patches
            rgb_global, nir_global, tir_global: [B, dim] 三个模态的 CLS 全局特征

        输出:
            final_feature: [B, 3*dim] 融合后的特征
        """

        # ============ 第1步：构建 Query 和 Key-Value ============
        # 将三个模态的 CLS token 堆叠作为 Query，用于"询问"选中的 patches
        # 公式 (16): Q̄ = T[f_R^c, f_N^c, f_T^c]
        queries = torch.stack([rgb_global, nir_global, tir_global], dim=1)  # [B, 3, dim]

        # 将三个模态选中的 patches 拼接作为 Key 和 Value，供 Query 查询
        # 公式 (16): K̄ = C[f̃_R^p, f̃_N^p, f̃_T^p]
        keys_values = torch.cat([rgb_selected, nir_selected, tir_selected], dim=1)  # [B, 3*L, dim]

        # ============ 第2步：多头交叉注意力 (MHCA) ============
        # Query (CLS tokens) 去关注 Key-Value (选中的 patches)，聚合重要信息
        # 对应论文公式 (17): MHCA(Q̄, K̄, K̄)
        attn_output, _ = self.cross_attn(queries, keys_values, keys_values)
        # attn_output: [B, 3, dim]

        # ============ 第3步：残差连接 + LayerNorm ============
        # 对应论文公式 (17): Q̄' = LN(Q̄ + MHCA(Q̄, K̄, K̄))
        attn_output = self.norm1(queries + attn_output)  # [B, 3, dim]

        # ============ 第4步：前馈神经网络 (FFN) ============
        # 标准 Transformer 的 FFN: Linear -> GELU -> Linear
        ffn_output = self.ffn(attn_output)  # [B, 3, dim]

        # ============ 第5步：第二次残差连接 + LayerNorm ============
        # 对应论文公式 (18): F̄ = LN(Q̄' + FFN(Q̄'))
        fused_features = self.norm2(attn_output + ffn_output)  # [B, 3, dim]

        # ============ 第6步：拼接三个模态的融合特征 ============
        # fused_features[:, 0]: [B, dim] RGB 的融合特征
        # fused_features[:, 1]: [B, dim] NIR 的融合特征
        # fused_features[:, 2]: [B, dim] TIR 的融合特征
        # 拼接得到最终特征
        final_feature = torch.cat([fused_features[:, 0], fused_features[:, 1], fused_features[:, 2]], dim=1)
        # final_feature: [B, 3*dim] = [B, 1536]

        return final_feature


class LayerNorm(nn.LayerNorm):
    """
    支持 FP16 的 LayerNorm
    在混合精度训练中，先转为 FP32 计算再转回原类型，确保数值稳定性
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Select_Interactive_Module(nn.Module):
    """
    SIM (Selective Interaction Module) - 选择性交互模块

    完整的 SIM 模块，包含:
    1. TokenSelection: Token 选择（模态内 + 模态间）
    2. ModalInteractive: 模态交互（交叉注意力融合）

    输入:
        rgb_patches, nir_patches, tir_patches: [B, 128, 512] 三模态 patch tokens
        rgb_global, nir_global, tir_global: [B, 512] 三模态 CLS tokens

    输出:
        final_feature: [B, 1536] 融合后的特征向量
    """

    def __init__(self, dim, k=112):
        """
        参数:
            dim: 特征维度 (512)
            k: Token 选择数量 (默认 112，实际使用 cfg.MODEL.TOPK)
        """
        super().__init__()
        num_heads = 8
        self.token_selection = TokenSelection(dim, k)
        self.modal_interactive = ModalInteractive(dim, num_heads)

    def forward(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
        """
        SIM 前向传播

        数据流:
        patches [B,128,512] x3 + globals [B,512] x3
          -> TokenSelection: 筛选重要 patches
          -> ModalInteractive: 交叉注意力融合
          -> final_feature [B, 1536]
        """
        # 1. Token 选择: 筛选重要的 patches (未选中的置为 0)
        rgb_selected, nir_selected, tir_selected = self.token_selection(
            rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global
        )
        # *_selected: [B, 128, 512]

        # 2. 模态交互: CLS tokens 通过交叉注意力聚合 patches 信息
        final_feature = self.modal_interactive(
            rgb_selected, nir_selected, tir_selected, rgb_global, nir_global, tir_global
        )
        # final_feature: [B, 1536]

        return final_feature
