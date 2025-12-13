"""
Signal 模型主文件
定义了 Signal 多模态重识别模型的核心架构
"""

import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
import torch
import torch.nn.functional as F


from modeling.AddModule.useA import Select_Interactive_Module  # SIM 选择性交互模块
from modeling.AddModule.useB import AlignmentM  # GAM+LAM 对齐模块


class Signal(nn.Module):
    """
    Signal 多模态目标重识别模型

    核心组件:
    - clip_vision_encoder: ViT backbone，提取各模态特征
    - SIM: 选择性交互模块，进行 token 选择和跨模态交互
    - AlignM: 对齐模块，包含 GAM (全局对齐) 和 LAM (局部对齐)

    输入: {'RGB': [B,3,256,128], 'NI': [B,3,256,128], 'TI': [B,3,256,128]}
    输出: 根据配置返回不同组合的 (score, features, losses)
    """

    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        """
        初始化 Signal 模型

        参数:
            num_classes: 类别数（身份数量）
            cfg: 配置对象
            camera_num: 相机数量（用于 camera embedding）
            view_num: 视角数量
            factory: backbone 工厂字典
        """
        super(Signal, self).__init__()

        # ============ 特征维度设置 ============
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768  # 标准 ViT-Base 特征维度
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512  # CLIP ViT-B-16 特征维度

        # ============ 基本配置 ============
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.direct = cfg.MODEL.DIRECT  # 1: 拼接三模态特征, 0: 分别处理
        self.camera = camera_num
        self.view = view_num
        self.use_A = cfg.MODEL.USE_A  # 是否使用 SIM
        self.use_B = cfg.MODEL.USE_B  # 是否使用 GAM+LAM

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN  # [256, 128]
        # patch 数量计算: 256/16=16, 128/16=8, 共 16*8=128 个 patches
        self.h, self.w = self.image_size[0]//16, self.image_size[1]//16  # h=16, w=8

        # ============ Backbone: CLIP ViT 编码器 ============
        # 输入: [B, 3, 256, 128] 单模态图像
        # 输出: (patch_tokens [B, 128, 512], cls_token [B, 512])
        self.clip_vision_encoder = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)

        # ============ 分类器和 BNNeck ============
        if self.direct:
            # 拼接模式: 三模态特征直接拼接 [B, 3*feat_dim]
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)  # BNNeck: [B, 1536]
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)  # [1536] -> [num_classes]
            self.classifier.apply(weights_init_classifier)
        else:
            # 分离模式: 每个模态单独分类器
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)

            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)
            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)

            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)

            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)

        # ============ SIM: 选择性交互模块 ============
        if self.use_A:
            num = 3  # 三个模态
            # SIM 输入: 三模态的 patches 和 globals
            # SIM 输出: 融合特征 [B, 3*feat_dim]
            self.SIM = Select_Interactive_Module(self.feat_dim, k=int(cfg.MODEL.TOPK))

            # SIM 输出的分类器
            self.classifier_var = nn.Linear(num*self.feat_dim, self.num_classes, bias=False)
            self.classifier_var.apply(weights_init_classifier)

            self.bottleneck_var = nn.BatchNorm1d(num*self.feat_dim)
            self.bottleneck_var.bias.requires_grad_(False)
            self.bottleneck_var.apply(weights_init_kaiming)

        # ============ AlignM: GAM + LAM 对齐模块 ============
        if self.use_B:
            # 输入: 三模态的 patch tokens
            # 输出: GAM loss (标量), LAM loss (标量)
            self.AlignM = AlignmentM(self.feat_dim, self.h, self.w)


    def load_param(self, trained_path):
        """加载预训练模型参数"""
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        """计算模型 FLOPs"""
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=1, training=True, sge="CLS"):
        """
        前向传播

        参数:
            x: 输入字典 {'RGB': [B,3,256,128], 'NI': [B,3,256,128], 'TI': [B,3,256,128]}
            label: 身份标签 [B]
            cam_label: 相机标签 [B]
            view_label: 视角标签 [B]
            training: 是否训练模式
            sge: 对齐阶段 "CLS" 或 "together_CLS_Patch"

        返回 (训练模式, direct=True):
            sign=1: (sign, ori_score [B,C], ori [B,1536])
            sign=2: (sign, ori_score, ori, vars_score [B,C], vars_total [B,1536])
            sign=3: (sign, ori_score, ori, vars_score, vars_total, loss_area, [patch_loss])

        返回 (推理模式):
            不使用SIM: ori [B, 1536]
            使用SIM: concat([ori, vars_total]) [B, 3072]
        """

        if training:
            # ============ 训练模式 ============

            # 1. 获取三模态输入图像
            RGB = x['RGB']  # [B, 3, 256, 128]
            NI = x['NI']    # [B, 3, 256, 128]
            TI = x['TI']    # [B, 3, 256, 128]

            # 2. 通过 ViT Backbone 提取特征
            # 输入: [B, 3, 256, 128]
            # 输出: patch_tokens [B, 128, 512], global_cls [B, 512]
            rgb_patch, RGB_global = self.clip_vision_encoder(RGB, cam_label=cam_label, view_label=view_label)
            ni_patch, NI_global = self.clip_vision_encoder(NI, cam_label=cam_label, view_label=view_label)
            ti_patch, TI_global = self.clip_vision_encoder(TI, cam_label=cam_label, view_label=view_label)
            # rgb_patch: [B, 128, 512] - 128个patch tokens
            # RGB_global: [B, 512] - CLS token 作为全局特征

            # 3. SIM: 选择性交互模块
            if self.use_A:
                # 输入: patches [B,128,512] x3, globals [B,512] x3
                # 输出: vars_total [B, 1536] 融合后的特征
                vars_total = self.SIM(rgb_patch, ni_patch, ti_patch, RGB_global, NI_global, TI_global)

                # BNNeck + 分类
                vars_global = self.bottleneck_var(vars_total)  # [B, 1536]
                vars_score = self.classifier_var(vars_global)   # [B, num_classes]

            # 4. AlignM: GAM + LAM 对齐损失
            if self.use_B:
                if sge == "CLS":
                    # 只计算 GAM loss
                    loss_area = self.AlignM(rgb_patch, ni_patch, ti_patch, stage="CLS")
                    # loss_area: 标量, GAM 3D多面体体积损失
                else:
                    # 计算 GAM + LAM loss
                    loss_area, patch_loss = self.AlignM(rgb_patch, ni_patch, ti_patch, stage="together_CLS_Patch")
                    # loss_area: GAM loss (标量)
                    # patch_loss: LAM loss (标量)

            # 5. 基线特征分类
            if self.direct:
                # 拼接三模态 CLS 特征
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)  # [B, 1536]
                ori_global = self.bottleneck(ori)  # BNNeck: [B, 1536]
                ori_score = self.classifier(ori_global)  # [B, num_classes]
            else:
                # 分别分类
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))

            # 6. 根据配置返回不同输出
            if self.direct:
                if not self.use_A:
                    # 基线模式: 只有原始特征
                    sign = 1
                    return sign, ori_score, ori

                elif self.use_A and self.use_B == False:
                    # +SIM: 原始特征 + SIM特征
                    sign = 2
                    return sign, ori_score, ori, vars_score, vars_total

                elif self.use_A and self.use_B:
                    # +SIM+GAM(+LAM): 完整模型
                    sign = 3
                    if sge == "CLS":
                        return sign, ori_score, ori, vars_score, vars_total, loss_area
                    else:
                        return sign, ori_score, ori, vars_score, vars_total, loss_area, patch_loss
            else:
                # 非 direct 模式 (分离各模态)
                if not self.use_A:
                    sign = 1
                    return sign, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global

                elif self.use_A and self.use_B == False:
                    sign = 2
                    return sign, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, vars_score, vars_total

                elif self.use_A and self.use_B:
                    sign = 3
                    if sge == "CLS":
                        return sign, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, vars_score, vars_total, loss_area
                    else:
                        return sign, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, vars_score, vars_total, loss_area, patch_loss

        else:
            # ============ 推理模式 ============

            RGB = x['RGB']  # [B, 3, 256, 128]
            NI = x['NI']
            TI = x['TI']

            if 'cam_label' in x:
                cam_label = x['cam_label']

            # 提取特征
            rgb_patch, RGB_global = self.clip_vision_encoder(RGB, cam_label=cam_label, view_label=view_label)
            ni_patch, NI_global = self.clip_vision_encoder(NI, cam_label=cam_label, view_label=view_label)
            ti_patch, TI_global = self.clip_vision_encoder(TI, cam_label=cam_label, view_label=view_label)

            # SIM 特征提取 (推理时不需要 loss)
            if self.use_A:
                vars_total = self.SIM(rgb_patch, ni_patch, ti_patch, RGB_global, NI_global, TI_global)

            # AlignM (推理时忽略返回值)
            if self.use_B:
                if sge == "CLS":
                    _ = self.AlignM(rgb_patch, ni_patch, ti_patch, stage="CLS")
                else:
                    _, _ = self.AlignM(rgb_patch, ni_patch, ti_patch, stage="together_CLS_Patch")

            # 拼接基线特征
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)  # [B, 1536]

            # 返回用于 ReID 匹配的特征
            if not self.use_A:
                return ori  # [B, 1536]
            else:
                return torch.cat([ori, vars_total], dim=-1)  # [B, 3072]


# ============ Backbone 工厂字典 ============
__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_frame(cfg, num_class, camera_num, view_num=0):
    """
    构建 Signal 模型的工厂函数

    参数:
        cfg: 配置对象
        num_class: 类别数
        camera_num: 相机数量
        view_num: 视角数量

    返回:
        Signal 模型实例
    """
    model = Signal(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building Signal===========')
    return model


class LayerNorm(nn.LayerNorm):
    """
    支持 FP16 的 LayerNorm
    在注意力前后使用，确保数值稳定性
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
