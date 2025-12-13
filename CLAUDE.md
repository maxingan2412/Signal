# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 提供本代码库的指导信息。

## 项目概述

Signal 是一个多模态目标重识别（ReID）研究框架，用于跨 RGB、近红外（NI）和热红外（TI）成像模态识别行人/车辆。发表于 AAAI-2026 (arXiv:2511.17965)。

## 构建与运行命令

```bash
# 训练
python train.py --config_file configs/RGBNT201/Signal.yml

# 测试/评估
python test.py --config_file configs/RGBNT201/Signal.yml

# 其他数据集: configs/RGBNT100/Signal.yml, configs/MSVR310/Signal.yml
```

## 架构

### 核心组件

1. **Backbone**: CLIP ViT-B-16 视觉Transformer，输出 patch tokens 和全局 CLS 特征

2. **SIM (选择性交互模块)** - `modeling/AddModule/useA.py`:
   - 模态内/模态间 token 选择，缓解背景干扰
   - 由 `cfg.MODEL.USE_A` 和 `cfg.MODEL.TOPK` 控制

3. **GAM (全局对齐模块)** - `modeling/AddModule/useB.py`:
   - 3D 多面体体积最小化，实现跨模态全局特征对齐
   - 损失权重: `cfg.MODEL.Gram_Loss_weight`

4. **LAM (局部对齐模块)** - `modeling/AddModule/useB.py` (`patch_Align`):
   - 可变形空间采样，实现像素级对齐
   - 损失权重: `cfg.MODEL.PAT_Loss_weight`

### 数据流

```
RGB, NI, TI 图像 (256x128)
  -> ViT 编码器 (每个模态)
    -> Patch tokens [batch, 256, feat_dim] + CLS token [batch, feat_dim]
      -> SIM: token 选择
      -> GAM: 全局对齐
      -> LAM: 局部对齐
        -> 分类 + 组合损失
```

### 模型输出签名

`Signal` 模型根据 "sign" 标志返回不同输出：
- `sign=1`: 基线输出 `(sign, ori_score, ori)`
- `sign=2`: +SIM `(sign, ori_score, ori, vars_score, vars_total)`
- `sign=3`: +SIM+GAM(+LAM) `(..., loss_area, [patch_loss])`

推理时: 如果启用 USE_A，返回拼接特征 `torch.cat([ori, vars_total])`。

## 配置

使用 YACS 配置系统。配置文件中的关键参数：

```yaml
MODEL:
  USE_A: True                    # 启用 SIM
  USE_B: True                    # 启用 GAM+LAM
  TOPK: 80                       # 选择的 token 数量
  stageName: 'together_CLS_Patch'  # 或 'CLS'
  ID_LOSS_WEIGHT: 0.25
  TRIPLET_LOSS_WEIGHT: 1.0
  Gram_Loss_weight: 0.2          # GAM 损失
  PAT_Loss_weight: 0.2           # LAM 损失

SOLVER:
  BASE_LR: 0.00035
  MAX_EPOCHS: 50
  IMS_PER_BATCH: 64
```

默认值定义在 `config/defaults.py`。

## 关键文件

| 文件 | 用途 |
|------|------|
| `train.py` | 训练入口 |
| `test.py` | 测试入口 |
| `engine/processor.py` | `do_train`, `do_inference` 主循环 |
| `modeling/make_model.py` | `Signal` 类定义 |
| `modeling/meta_arch.py` | `build_transformer` backbone 工厂 |
| `modeling/AddModule/useA.py` | SIM 实现 |
| `modeling/AddModule/useB.py` | GAM + LAM 实现 |
| `data/datasets/make_dataloader.py` | DataLoader 工厂 |
| `layers/make_loss.py` | 损失函数工厂 |
| `utils/metrics.py` | `R1_mAP_eval` 评估 |

## 支持的数据集

- **RGBNT201**: 201 个身份, RGB + NI + TI
- **RGBNT100**: 100 个身份 (较小)
- **MSVR310**: 多场景车辆 ReID, 310 个身份

数据集结构需要 `train/` 和 `test/` 目录，每个身份包含多模态图像变体。

## 开发工作流

### 验证代码修改

**重要**: 修改任何模块后，始终通过运行实际训练脚本来验证：

```bash
# 使用 Signal_test.yml 进行快速验证（较少 epoch，较小 batch）
python train.py --config_file configs/RGBNT201/Signal_test.yml
```

**不要**为单个文件编写独立测试脚本。始终通过完整训练流程测试，确保模块正确集成。

### 理解代码执行流程

调试或修改代码时，使用训练日志和系统输出来理解实际执行的代码路径。日志中的真实执行上下文应指导你的理解：

- 训练日志显示初始化了哪些模型组件
- 损失计算日志揭示哪些模块处于激活状态（`USE_A`, `USE_B` 标志）
- 错误回溯显示实际调用栈

**关键原则**: 使用实际执行的代码路径作为真正的上下文，而非基于文件结构的假设。这避免了对模块交互方式的误导性解读。

### 执行流程参考

```
train.py
  -> 配置加载 (config/defaults.py + yaml)
  -> make_dataloader() (data/datasets/make_dataloader.py)
  -> make_model() (modeling/make_model.py) -> Signal 类
     -> build_transformer() (modeling/meta_arch.py) -> ViT backbone
     -> Select_Interactive_Module (modeling/AddModule/useA.py) 如果 USE_A
     -> AlignmentM (modeling/AddModule/useB.py) 如果 USE_B
  -> make_loss() (layers/make_loss.py)
  -> make_optimizer() (solver/make_optimizer.py)
  -> do_train() (engine/processor.py) -> 主训练循环
```

## 执行追踪

追踪代码执行流程（类似 PyCharm Debug）：

```bash
# 追踪函数调用（输出调用树）
python scripts/trace_execution.py --mode calls --config configs/RGBNT201/Signal_test.yml

# 只追踪特定模块
python scripts/trace_execution.py --mode calls --filter modeling --config configs/RGBNT201/Signal_test.yml

# 保存到文件
python scripts/trace_execution.py --mode calls --output trace_log.txt --config configs/RGBNT201/Signal_test.yml
```

## 评估指标

- **mAP**: 平均精度均值（主要指标）
- **Rank-1/5/10**: CMC 曲线指标
- 每 `cfg.SOLVER.EVAL_PERIOD` 个 epoch 在验证时计算

## 代码模式

- 输入格式: `img = {'RGB': tensor, 'NI': tensor, 'TI': tensor}`，形状 `[batch, 3, 256, 128]`
- 默认启用混合精度训练 (AMP)
- 多 GPU 通过 DDP: `cfg.MODEL.DIST_TRAIN = True`
- 检查点每 `cfg.SOLVER.CHECKPOINT_PERIOD` 个 epoch 保存到 `cfg.OUTPUT_DIR`
