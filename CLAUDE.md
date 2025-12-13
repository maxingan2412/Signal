# CLAUDE.md

Signal 多模态目标重识别框架指导文件。

## 快速命令

```bash
# 训练
python train.py --config_file configs/RGBNT201/Signal.yml

# 测试
python test.py --config_file configs/RGBNT201/Signal.yml

# 快速验证（修改代码后）
python train.py --config_file configs/RGBNT201/Signal_test.yml

# 执行追踪
python scripts/trace_execution.py --mode calls --config configs/RGBNT201/Signal_test.yml
```

## 核心架构

```
输入: RGB, NI, TI 图像 [B, 3, 256, 128]
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  CLIP ViT-B-16 Backbone (每模态独立编码)                 │
│  输出: patch_tokens [B, 128, 512] + CLS [B, 512]        │
└─────────────────────────────────────────────────────────┘
  │
  ├── USE_A=True ──▶ SIM (选择性交互模块)
  │                   ├─ TokenSelection: 模态内+模态间 token 选择
  │                   └─ ModalInteractive: 交叉注意力融合 → [B, 1536]
  │
  ├── USE_B=True ──▶ GAM (全局对齐): 3D多面体体积最小化 → loss_area
  │              └─▶ LAM (局部对齐): 可变形采样+MSE → patch_loss
  │
  ▼
输出特征: [B, 1536] (基线) 或 [B, 3072] (基线+SIM)
```

## 完整执行流程

```
train.py
  │
  ├─▶ config 加载
  │     config/defaults.py (默认值)
  │     configs/RGBNT201/Signal.yml (覆盖)
  │
  ├─▶ make_dataloader() ─────────────────────────────────────────────┐
  │     data/datasets/make_dataloader.py:185                         │
  │       └─▶ RGBNT201() → data/datasets/RGBNT201.py:11              │
  │       └─▶ RandomIdentitySampler() → data/datasets/sampler.py:18  │
  │                                                                   │
  ├─▶ make_frame() ──────────────────────────────────────────────────┤
  │     modeling/make_model.py:303                                    │
  │       └─▶ Signal.__init__() :35                                  │
  │             │                                                     │
  │             ├─▶ build_transformer()                              │
  │             │     modeling/meta_arch.py:34                        │
  │             │       └─▶ load_clip_to_cpu() :177                  │
  │             │             modeling/clip/model.py:651 build_model  │
  │             │               └─▶ VisionTransformer :419            │
  │             │                     └─▶ Transformer :407            │
  │             │                           └─▶ ResidualAttentionBlock│
  │             │                                                     │
  │             ├─▶ Select_Interactive_Module() [USE_A=True]         │
  │             │     modeling/AddModule/useA.py:358                  │
  │             │       ├─▶ TokenSelection :17                        │
  │             │       └─▶ ModalInteractive :260                     │
  │             │                                                     │
  │             └─▶ AlignmentM() [USE_B=True]                        │
  │                   modeling/AddModule/useB.py:28                   │
  │                     └─▶ DAS (DA_sample) x3                        │
  │                           modeling/AddModule/DAS.py:17            │
  │                                                                   │
  ├─▶ make_loss() ───────────────────────────────────────────────────┤
  │     layers/make_loss.py:29                                        │
  │       ├─▶ CenterLoss → layers/center_loss.py:7                   │
  │       ├─▶ TripletLoss → layers/triplet_loss.py:107               │
  │       └─▶ CrossEntropyLabelSmooth → layers/softmax_loss.py:4     │
  │                                                                   │
  ├─▶ make_optimizer() ──────────────────────────────────────────────┤
  │     solver/make_optimizer.py:4                                    │
  │                                                                   │
  ├─▶ create_scheduler() ────────────────────────────────────────────┤
  │     solver/scheduler_factory.py:7                                 │
  │       └─▶ CosineLRScheduler → solver/cosine_lr.py:17             │
  │                                                                   │
  └─▶ do_train() ────────────────────────────────────────────────────┘
        engine/processor.py:41

        训练循环 (每个 batch):
        ┌────────────────────────────────────────────────────────────┐
        │ img = {'RGB': [B,3,256,128], 'NI': ..., 'TI': ...}        │
        │                        │                                   │
        │                        ▼                                   │
        │ model.forward(training=True, sge=stage)                    │
        │   modeling/make_model.py:147                               │
        │     │                                                      │
        │     ├─▶ clip_vision_encoder(RGB/NI/TI)                    │
        │     │     → patch [B,128,512], global [B,512]              │
        │     │                                                      │
        │     ├─▶ SIM() [USE_A]                                     │
        │     │     modeling/AddModule/useA.py:385                   │
        │     │       ├─▶ token_selection.forward()                  │
        │     │       │     ├─▶ inter_modal_token_selection :96     │
        │     │       │     └─▶ intra_modal_token_selection :48     │
        │     │       └─▶ modal_interactive.forward() :296          │
        │     │     → vars_total [B, 1536]                           │
        │     │                                                      │
        │     └─▶ AlignM() [USE_B]                                  │
        │           modeling/AddModule/useB.py:169                   │
        │             ├─▶ Cls_Align() :76  → loss_area (GAM)        │
        │             │     └─▶ volume_computation3()                │
        │             └─▶ patch_Align() :128  → patch_loss (LAM)    │
        │                   └─▶ DAS_r/n/t.forward()                  │
        │                         modeling/AddModule/DAS.py:107      │
        │                                                            │
        │ 返回: (sign, ori_score, ori, vars_score, vars_total,      │
        │        loss_area, [patch_loss])                            │
        │                        │                                   │
        │                        ▼                                   │
        │ loss_fn(score, feat, target)                               │
        │   layers/make_loss.py:109                                  │
        │     → ID_LOSS (CrossEntropy) + TRI_LOSS (Triplet)          │
        │                        │                                   │
        │                        ▼                                   │
        │ total_loss = loss + α*loss_area + β*patch_loss             │
        │   (α=Gram_Loss_weight, β=PAT_Loss_weight)                  │
        │                        │                                   │
        │                        ▼                                   │
        │ scaler.scale(loss).backward()                              │
        │ scaler.step(optimizer)                                     │
        └────────────────────────────────────────────────────────────┘

        验证 (每 eval_period 个 epoch):
        └─▶ training_neat_eval() :249
              └─▶ model.forward(training=False)
                    → feat [B, 3072]
              └─▶ evaluator.compute() → mAP, CMC
                    utils/metrics.py:222
```

## 关键文件定位

| 功能 | 文件:行号 |
|------|----------|
| 模型主类 | `modeling/make_model.py:22` Signal |
| ViT Backbone | `modeling/meta_arch.py:34` build_transformer |
| SIM 模块 | `modeling/AddModule/useA.py:358` Select_Interactive_Module |
| Token 选择 | `modeling/AddModule/useA.py:17` TokenSelection |
| 模态交互 | `modeling/AddModule/useA.py:260` ModalInteractive |
| GAM 对齐 | `modeling/AddModule/useB.py:76` Cls_Align |
| LAM 对齐 | `modeling/AddModule/useB.py:128` patch_Align |
| 可变形采样 | `modeling/AddModule/DAS.py:17` DA_sample |
| 训练循环 | `engine/processor.py:41` do_train |
| 损失函数 | `layers/make_loss.py:29` make_loss |
| 评估指标 | `utils/metrics.py:222` R1_mAP_eval |

## 配置参数

```yaml
MODEL:
  USE_A: True           # SIM 模块
  USE_B: True           # GAM+LAM 模块
  TOPK: 80              # Token 选择数量
  stageName: 'together_CLS_Patch'  # 'CLS' 或 'together_CLS_Patch'
  ID_LOSS_WEIGHT: 0.25
  TRIPLET_LOSS_WEIGHT: 1.0
  Gram_Loss_weight: 0.2   # GAM 损失权重
  PAT_Loss_weight: 0.2    # LAM 损失权重
```

## 模型输出格式

```python
# 训练时 (sign 标识输出类型)
sign=1: (sign, ori_score [B,C], ori [B,1536])                    # 基线
sign=2: (sign, ori_score, ori, vars_score, vars_total)           # +SIM
sign=3: (..., loss_area)                       # +SIM+GAM (stage=CLS)
sign=3: (..., loss_area, patch_loss)           # +SIM+GAM+LAM (stage=together_CLS_Patch)

# 推理时
USE_A=False: ori [B, 1536]
USE_A=True:  torch.cat([ori, vars_total]) [B, 3072]
```

## 开发原则

1. **验证修改**: 改代码后运行 `python train.py --config_file configs/RGBNT201/Signal_test.yml`
2. **理解执行流**: 用日志和追踪脚本确认实际执行路径，而非假设
3. **定位代码**: 参考上方执行流程图和文件定位表快速找到目标代码
