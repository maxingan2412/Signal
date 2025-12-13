# encoding: utf-8
"""
损失函数工厂模块

构建 Signal 模型训练所需的损失函数:
1. ID Loss (CrossEntropy): 身份分类损失
2. Triplet Loss: 度量学习损失，拉近同类、推远异类
3. Center Loss: 类内紧凑性损失 (可选)

总损失公式:
    L_total = α * L_ID + β * L_Triplet + (γ * L_GAM + δ * L_LAM)

其中:
    α = cfg.MODEL.ID_LOSS_WEIGHT (默认 0.25)
    β = cfg.MODEL.TRIPLET_LOSS_WEIGHT (默认 1.0)
    γ = cfg.MODEL.Gram_Loss_weight (默认 0.2，在 processor.py 中应用)
    δ = cfg.MODEL.PAT_Loss_weight (默认 0.2，在 processor.py 中应用)

@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):
    """
    损失函数工厂

    参数:
        cfg: 配置对象 (YACS)
        num_classes: 类别数 (身份数量)

    返回:
        loss_func: 损失函数，签名为 loss_func(score, feat, target, target_cam)
            - score: 分类 logits [B, num_classes]
            - feat: 特征向量 [B, feat_dim] 用于 triplet loss
            - target: 身份标签 [B]
            - target_cam: 相机标签 [B] (当前未使用)
        center_criterion: 中心损失模块 (CenterLoss)

    配置参数:
        cfg.DATALOADER.SAMPLER: 采样策略 ('softmax', 'softmax_triplet')
        cfg.MODEL.METRIC_LOSS_TYPE: 度量损失类型 ('triplet')
        cfg.MODEL.IF_LABELSMOOTH: 是否使用标签平滑 ('on'/'off')
        cfg.MODEL.NO_MARGIN: 是否使用 soft triplet loss
        cfg.SOLVER.MARGIN: triplet loss 的 margin 值 (默认 0.3)
        cfg.MODEL.ID_LOSS_WEIGHT: ID loss 权重 (默认 0.25)
        cfg.MODEL.TRIPLET_LOSS_WEIGHT: Triplet loss 权重 (默认 1.0)
    """
    sampler = cfg.DATALOADER.SAMPLER  # 'softmax_triplet'

    # ============ 中心损失 (Center Loss) ============
    # 目的: 增强特征的类内紧凑性和类间分离性
    # 为每个类别学习一个特征中心，最小化样本到其类中心的距离
    feat_dim = 2048  # 注意: 实际模型使用 512 或 1536，此处可能需要调整
    center_criterion = CenterLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        use_gpu=False
    )

    # ============ Triplet Loss 配置 ============
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            # Soft Triplet Loss: 不使用固定 margin，而是用 softplus 函数
            # L = log(1 + exp(d_ap - d_an))
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            # Hard Triplet Loss with margin
            # L = max(0, d_ap - d_an + margin)
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    # ============ ID Loss (CrossEntropy) 配置 ============
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # 标签平滑: 防止过拟合，提高泛化能力
        # 将 one-hot 标签 [1, 0, 0, ...] 变为 [0.9, 0.1/K, 0.1/K, ...]
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    # ============ 损失函数定义 ============
    if sampler == 'softmax':
        # 纯分类模式: 只使用 CrossEntropy 损失
        def loss_func(score, feat, target, target_cam):
            """
            纯 softmax 损失

            参数:
                score: [B, num_classes] 分类 logits
                feat: [B, feat_dim] 特征 (未使用)
                target: [B] 身份标签
                target_cam: [B] 相机标签 (未使用)

            返回:
                loss: 标量，CrossEntropy 损失
            """
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        # 联合训练模式: ID Loss + Triplet Loss
        def loss_func(score, feat, target, target_cam):
            """
            Softmax + Triplet 联合损失

            参数:
                score: [B, num_classes] 或 list[[B, num_classes], ...] 分类 logits
                feat: [B, feat_dim] 或 list[[B, feat_dim], ...] 特征向量
                target: [B] 身份标签
                target_cam: [B] 相机标签 (当前未使用)

            返回:
                loss: 标量，加权组合损失
                    = ID_LOSS_WEIGHT * ID_LOSS + TRIPLET_LOSS_WEIGHT * TRI_LOSS

            数据流:
                score [B, C] -> CrossEntropy -> ID_LOSS (标量)
                feat [B, D] -> TripletLoss -> TRI_LOSS (标量)
                -> 加权求和 -> total_loss

            当 score/feat 是 list 时 (多特征输出):
                分别计算每个特征的损失，然后取加权平均
                第一个特征权重 0.5，其余特征权重 0.5
            """
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # ============ 带标签平滑的损失计算 ============

                    # ID Loss (CrossEntropy with Label Smoothing)
                    if isinstance(score, list):
                        # 多特征情况: score = [score0, score1, score2, ...]
                        # score[1:] 的损失取平均，然后与 score[0] 各占 50%
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        # 单特征情况
                        ID_LOSS = xent(score, target)  # 标量

                    # Triplet Loss
                    if isinstance(feat, list):
                        # 多特征情况: feat = [feat0, feat1, feat2, ...]
                        # triplet 返回 (loss, ap_dist, an_dist)，取 [0] 获取 loss
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        # 单特征情况
                        TRI_LOSS = triplet(feat, target)[0]  # 标量

                    # 加权组合
                    # 默认: 0.25 * ID_LOSS + 1.0 * TRI_LOSS
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                else:
                    # ============ 不带标签平滑的损失计算 ============

                    # ID Loss (标准 CrossEntropy)
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)  # 标量

                    # Triplet Loss
                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]  # 标量

                    # 加权组合
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet '
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func, center_criterion
