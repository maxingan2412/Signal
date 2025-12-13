"""
Signal 模型训练与推理处理器

包含三个核心函数:
1. do_train: 主训练循环
2. do_inference: 测试推理
3. training_neat_eval: 训练中的验证评估
"""

import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
import math


def adjust_weights(epoch, total_epochs, initial_weight, final_weight):
    """
    动态调整损失权重 (线性预热)

    参数:
        epoch: 当前 epoch
        total_epochs: 总 epoch 数
        initial_weight: 初始权重
        final_weight: 最终权重

    返回:
        调整后的权重值
    """
    # 在前 20% 的 epoch 内线性增加权重
    alpha = min(epoch / (total_epochs * 0.2), 1.0)
    return initial_weight + alpha * (final_weight - initial_weight)


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, stage):
    """
    Signal 模型主训练循环

    参数:
        cfg: 配置对象 (YACS)
        model: Signal 模型实例
        center_criterion: 中心损失 (CenterLoss)
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 主优化器 (Adam/SGD)
        optimizer_center: 中心损失优化器
        scheduler: 学习率调度器 (CosineLRScheduler)
        loss_fn: 损失函数 (包含 ID loss + Triplet loss)
        num_query: 验证集 query 数量
        local_rank: GPU 设备编号
        stage: 对齐阶段 "CLS" 或 "together_CLS_Patch"

    训练数据流:
        img [B, 3, 256, 128] x 3 模态
          -> model.forward(training=True)
            -> sign=1: (sign, ori_score [B,C], ori [B,1536])
            -> sign=2: (sign, ori_score, ori, vars_score [B,C], vars_total [B,1536])
            -> sign=3: (sign, ori_score, ori, vars_score, vars_total, loss_area, [patch_loss])
          -> loss_fn: ID loss + Triplet loss
          -> 反向传播 + 优化器更新
    """

    # ============ 训练参数配置 ============
    log_period = cfg.SOLVER.LOG_PERIOD          # 日志打印间隔 (迭代数)
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD  # 模型保存间隔 (epoch)
    eval_period = cfg.SOLVER.EVAL_PERIOD        # 验证间隔 (epoch)
    alpha = cfg.MODEL.Gram_Loss_weight          # GAM 损失权重 (默认 0.2)
    beta = cfg.MODEL.PAT_Loss_weight            # LAM 损失权重 (默认 0.2)

    # ============ 设备设置 ============
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    epochs = cfg.SOLVER.MAX_EPOCHS              # 总训练轮数

    # ============ 日志配置 ============
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("Signal.train")
    logger.info('start training')

    # ============ 模型部署到 GPU ============
    if device:
        print(f"use CUDA {device}")
        model.to(device)

        # 分布式训练配置
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device],
                find_unused_parameters=True  # 允许未使用的参数 (某些分支可能不激活)
            )

    # ============ 训练指标追踪器 ============
    loss_meter = AverageMeter()  # 损失均值追踪
    acc_meter = AverageMeter()   # 准确率均值追踪

    # ============ 评估器配置 ============
    # MSVR310 数据集使用不同的评估器 (处理场景 ID)
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    # ============ 混合精度训练 (AMP) ============
    scaler = amp.GradScaler()  # 梯度缩放器，防止 FP16 下溢

    # ============ 最佳指标记录 ============
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Start Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # ============ 主训练循环 ============
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # 重置指标追踪器
        loss_meter.reset()
        acc_meter.reset()

        # 更新学习率
        scheduler.step(epoch)

        # 设置模型为训练模式
        model.train()

        # ============ 批次迭代 ============
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            """
            数据格式:
                img: 字典 {'RGB': [B,3,256,128], 'NI': [B,3,256,128], 'TI': [B,3,256,128]}
                vid: 身份标签 [B]
                target_cam: 相机标签 [B]
                target_view: 视角标签 [B]
            """

            # 清零梯度
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            # ============ 数据转移到 GPU ============
            img = {
                'RGB': img['RGB'].to(device),  # [B, 3, 256, 128]
                'NI': img['NI'].to(device),    # [B, 3, 256, 128]
                'TI': img['TI'].to(device)     # [B, 3, 256, 128]
            }
            target = vid.to(device)            # [B] 身份标签
            target_cam = target_cam.to(device)  # [B] 相机标签
            target_view = target_view.to(device)  # [B] 视角标签

            # ============ 前向传播 (混合精度) ============
            with amp.autocast(enabled=True):
                """
                模型输出根据配置不同:
                - sign=1 (基线): (sign, ori_score [B,C], ori [B,1536])
                - sign=2 (+SIM): (sign, ori_score, ori, vars_score [B,C], vars_total [B,1536])
                - sign=3 (+SIM+GAM): stage="CLS" 时 (..., loss_area)
                                     stage="together_CLS_Patch" 时 (..., loss_area, patch_loss)
                """
                output = model(img, label=target, cam_label=target_cam,
                               view_label=target_view, training=True, sge=stage)

                loss = 0
                sign = output[0]  # 输出标识符 (1, 2, 或 3)

                # ============ 损失计算 - 基线模式 (sign=1) ============
                if sign == 1:
                    """
                    输出: (1, ori_score [B,C], ori [B,1536])
                    或 direct=False: (1, RGB_score, RGB_feat, NI_score, NI_feat, TI_score, TI_feat)

                    遍历 (score, feat) 对计算 ID loss + Triplet loss
                    """
                    index = len(output) - 1
                    for i in range(1, index, 2):
                        # output[i]: score [B, num_classes]
                        # output[i+1]: feat [B, feat_dim]
                        loss_tmp = loss_fn(
                            score=output[i],
                            feat=output[i + 1],
                            target=target,
                            target_cam=target_cam
                        )
                        loss = loss + loss_tmp

                # ============ 损失计算 - +SIM 模式 (sign=2) ============
                elif sign == 2:
                    """
                    输出: (2, ori_score, ori, vars_score, vars_total)
                    或 direct=False: (2, RGB_score, RGB_feat, ..., vars_score, vars_total)

                    包含基线特征和 SIM 融合特征的损失
                    """
                    index = len(output) - 1
                    for i in range(1, index, 2):
                        loss_tmp = loss_fn(
                            score=output[i],
                            feat=output[i + 1],
                            target=target,
                            target_cam=target_cam
                        )
                        loss = loss + loss_tmp

                # ============ 损失计算 - +SIM+GAM(+LAM) 模式 (sign=3) ============
                else:
                    """
                    输出 (stage="CLS"):
                        (3, ori_score, ori, vars_score, vars_total, loss_area)

                    输出 (stage="together_CLS_Patch"):
                        (3, ori_score, ori, vars_score, vars_total, loss_area, patch_loss)
                    """
                    if stage == "CLS":
                        # 只有 GAM 损失 (loss_area)
                        index = len(output) - 2  # 跳过最后的 loss_area
                        for i in range(1, index, 2):
                            loss_tmp = loss_fn(
                                score=output[i],
                                feat=output[i + 1],
                                target=target,
                                target_cam=target_cam
                            )
                            loss = loss + loss_tmp

                        # GAM 损失: 3D 多面体体积最小化
                        CLS_loss = output[-1]  # 标量
                        loss = loss + alpha * CLS_loss

                    else:
                        # GAM + LAM 损失
                        index = len(output) - 3  # 跳过最后的 loss_area 和 patch_loss
                        for i in range(1, index, 2):
                            loss_tmp = loss_fn(
                                score=output[i],
                                feat=output[i + 1],
                                target=target,
                                target_cam=target_cam
                            )
                            loss = loss + loss_tmp

                        # GAM 损失 (全局对齐) + LAM 损失 (局部对齐)
                        CLS_loss, pat_loss = output[-2], output[-1]  # 两个标量
                        loss = loss + alpha * CLS_loss + beta * pat_loss

            # ============ 反向传播 (混合精度) ============
            scaler.scale(loss).backward()  # 缩放损失后反向传播
            scaler.step(optimizer)         # 更新主优化器
            scaler.update()                # 更新缩放因子

            # ============ 中心损失优化器更新 ============
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                # 中心损失梯度缩放 (防止中心更新过快)
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # ============ 计算训练准确率 ============
            if isinstance(output, list):
                # output[1] 是第一个 score [B, num_classes]
                # output[1][0] 取第一个元素 (兼容某些输出格式)
                acc = (output[1][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[1].max(1)[1] == target).float().mean()

            # 更新指标追踪器
            loss_meter.update(loss.item(), img['RGB'].shape[0])  # 按 batch size 加权
            acc_meter.update(acc, 1)

            # ============ 日志输出 ============
            torch.cuda.synchronize()  # 确保 GPU 操作完成
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0])
                )

        # ============ Epoch 结束统计 ============
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch)
            )

        # ============ 模型保存 ============
        new_output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.ckpt_save_path)
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)

        # 定期保存检查点
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:  # 只在主进程保存
                    torch.save(
                        model.state_dict(),
                        os.path.join(new_output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                    )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(new_output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                )

        # ============ 定期验证 ============
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                mAP, cmc = training_neat_eval(
                    cfg, model, val_loader, device, evaluator, epoch, logger, sge=stage
                )

                # 更新最佳指标并保存最佳模型
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(
                        model.state_dict(),
                        os.path.join(new_output_dir, cfg.MODEL.NAME + 'best.pth')
                    )

                # 打印最佳指标
                logger.info("~" * 50)
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 logger,
                 sge,
                 local_rank):
    """
    Signal 模型推理/测试

    参数:
        cfg: 配置对象
        model: Signal 模型实例
        val_loader: 测试数据加载器
        num_query: query 图像数量 (用于划分 query/gallery)
        logger: 日志记录器
        sge: 对齐阶段 ("CLS" 或 "together_CLS_Patch")
        local_rank: GPU 设备编号

    推理数据流:
        img [B, 3, 256, 128] x 3 模态
          -> model.forward(training=False)
            -> USE_A=False: ori [B, 1536] (三模态 CLS 拼接)
            -> USE_A=True: concat([ori, vars_total]) [B, 3072]
          -> evaluator.update(): 累积特征
          -> evaluator.compute(): 计算 mAP 和 CMC
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("Signal.test")
    logger.info("Enter inferencing")

    # ============ 评估器配置 ============
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()

    # 模型部署到 GPU
    if device:
        model.to(device)

    # 设置为评估模式 (关闭 Dropout, BatchNorm 使用运行统计)
    model.eval()

    img_path_list = []
    logger.info("~" * 50)

    # ============ 推理循环 ============
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        """
        数据格式:
            img: 字典 {'RGB': [B,3,256,128], 'NI': [B,3,256,128], 'TI': [B,3,256,128]}
            pid: 身份标签 [B]
            camid: 相机 ID [B] (字符串形式)
            camids: 相机标签 [B] (数值形式)
            target_view: 视角/场景标签 [B]
            imgpath: 图像路径列表
        """
        with torch.no_grad():  # 推理时不计算梯度
            # 数据转移到 GPU
            img = {
                'RGB': img['RGB'].to(device),  # [B, 3, 256, 128]
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            camids = camids.to(device)
            scenceids = target_view  # 场景 ID (MSVR310 使用)
            target_view = target_view.to(device)

            # ============ 特征提取 ============
            # 输出: [B, 1536] 或 [B, 3072] (取决于 USE_A)
            feat = model(img, cam_label=camids, view_label=target_view,
                         training=False, sge=sge)

            # 更新评估器 (累积 query 和 gallery 特征)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))

            img_path_list.extend(imgpath)

    # ============ 计算评估指标 ============
    # 返回: CMC 曲线, mAP, 以及其他中间结果
    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    # ============ 输出结果 ============
    print("Validation Results ")
    print("mAP: {:.1%}".format(mAP))
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))

    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return cmc[0], cmc[4]  # 返回 Rank-1 和 Rank-5


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1, sge="CLS"):
    """
    训练过程中的验证评估

    与 do_inference 类似，但:
    1. 复用已有的 evaluator (避免重复创建)
    2. 返回 mAP 和完整 CMC 曲线用于最佳模型选择

    参数:
        cfg: 配置对象
        model: Signal 模型实例
        val_loader: 验证数据加载器
        device: 计算设备
        evaluator: R1_mAP_eval 评估器实例
        epoch: 当前 epoch (用于日志)
        logger: 日志记录器
        return_pattern: 输出模式 (保留参数，未使用)
        sge: 对齐阶段

    返回:
        mAP: 平均精度 (float)
        cmc: CMC 曲线 (numpy array, 长度 50)
    """
    # 重置评估器状态
    evaluator.reset()

    # 设置为评估模式
    model.eval()

    logger.info("~" * 50)

    # 提示当前使用的特征类型
    if not cfg.MODEL.USE_A:
        logger.info("Current is the base feature testing!")
        # 基线特征: 三模态 CLS 拼接 [B, 1536]
    else:
        logger.info("Current is the our feature testing!")
        # 完整特征: 基线 + SIM 融合 [B, 3072]

    logger.info("~" * 50)

    # ============ 验证循环 ============
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        """
        数据格式同 do_inference
        """
        with torch.no_grad():
            img = {
                'RGB': img['RGB'].to(device),  # [B, 3, 256, 128]
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)

            # 特征提取
            # 输出: [B, 1536] 或 [B, 3072]
            feat = model(img, cam_label=camids, view_label=target_view,
                         return_pattern=return_pattern, training=False, sge=sge)

            # 更新评估器
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))

    # ============ 计算指标 ============
    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    # ============ 输出验证结果 ============
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))

    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    logger.info("~" * 50)

    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    return mAP, cmc
