#!/bin/bash
# ============================================================================
# Signal Ablation Experiments - RGBNT201
# 测试 4 种模型配置的消融实验
# sign=1: 基线 (无 SIM, 无 GAM/LAM)
# sign=2: +SIM (有 SIM, 无 GAM/LAM)
# sign=3: +SIM+GAM (有 SIM, 有 GAM, stage=CLS)
# sign=3: +SIM+GAM+LAM (有 SIM, 有 GAM+LAM, stage=together_CLS_Patch)
# 4个实验，4个GPU并行
# ============================================================================
# 用法:
#   bash run_ablation_signal.sh [实验标识]
#
# 示例:
#   bash run_ablation_signal.sh 实验1
#   bash run_ablation_signal.sh ablation_v1
#   bash run_ablation_signal.sh  # 默认无标识
# ============================================================================

# 获取命令行参数（可选的实验标识）
EXP_TAG="${1:-}"

# Config file
CONFIG_FILE="configs/RGBNT201/Signal.yml"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 构建实验目录名（包含可选标识）
if [ -n "$EXP_TAG" ]; then
    EXP_DIR="logs/Signal_ablation_${EXP_TAG}_${TIMESTAMP}"
else
    EXP_DIR="logs/Signal_ablation_${TIMESTAMP}"
fi
mkdir -p ${EXP_DIR}

echo "=============================================================================="
echo "Signal Ablation Experiments - RGBNT201"
echo "=============================================================================="
if [ -n "$EXP_TAG" ]; then
    echo "实验标识: ${EXP_TAG}"
fi
echo "日志目录: ${EXP_DIR}"
echo "=============================================================================="
echo ""
echo "实验配置:"
echo "  sign=1: Baseline (USE_A=False, USE_B=False)"
echo "  sign=2: +SIM (USE_A=True, USE_B=False)"
echo "  sign=3: +SIM+GAM (USE_A=True, USE_B=True, stage=CLS)"
echo "  sign=3: +SIM+GAM+LAM (USE_A=True, USE_B=True, stage=together_CLS_Patch)"
echo "=============================================================================="
echo ""

# ============================================================================
# 4 个消融实验 (4 GPUs in parallel)
# ============================================================================
echo "Starting 4 ablation experiments on GPUs 0-3..."
echo ""

# GPU 0: sign=1 Baseline (无 SIM, 无 GAM/LAM)
# 输出: (sign, ori_score [B,C], ori [B,1536])
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config_file ${CONFIG_FILE} \
    MODEL.USE_A False \
    MODEL.USE_B False \
    OUTPUT_DIR "${EXP_DIR}/sign1_baseline" \
    > ${EXP_DIR}/sign1_baseline.log 2>&1 &
PID0=$!
echo "  GPU 0: sign=1 Baseline (USE_A=False, USE_B=False)"
echo "         PID: ${PID0}"
echo "         输出: (sign, ori_score [B,C], ori [B,1536])"
echo ""

# GPU 1: sign=2 +SIM (有 SIM, 无 GAM/LAM)
# 输出: (sign, ori_score, ori, vars_score, vars_total)
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config_file ${CONFIG_FILE} \
    MODEL.USE_A True \
    MODEL.USE_B False \
    OUTPUT_DIR "${EXP_DIR}/sign2_SIM" \
    > ${EXP_DIR}/sign2_SIM.log 2>&1 &
PID1=$!
echo "  GPU 1: sign=2 +SIM (USE_A=True, USE_B=False)"
echo "         PID: ${PID1}"
echo "         输出: (sign, ori_score, ori, vars_score, vars_total)"
echo ""

# GPU 2: sign=3 +SIM+GAM (有 SIM, 有 GAM, stage=CLS)
# 输出: (..., loss_area)
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config_file ${CONFIG_FILE} \
    MODEL.USE_A True \
    MODEL.USE_B True \
    MODEL.stageName "CLS" \
    OUTPUT_DIR "${EXP_DIR}/sign3_SIM_GAM" \
    > ${EXP_DIR}/sign3_SIM_GAM.log 2>&1 &
PID2=$!
echo "  GPU 2: sign=3 +SIM+GAM (USE_A=True, USE_B=True, stage=CLS)"
echo "         PID: ${PID2}"
echo "         输出: (..., loss_area)"
echo ""

# GPU 3: sign=3 +SIM+GAM+LAM (有 SIM, 有 GAM+LAM, stage=together_CLS_Patch)
# 输出: (..., loss_area, patch_loss)
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config_file ${CONFIG_FILE} \
    MODEL.USE_A True \
    MODEL.USE_B True \
    MODEL.stageName "together_CLS_Patch" \
    OUTPUT_DIR "${EXP_DIR}/sign3_SIM_GAM_LAM" \
    > ${EXP_DIR}/sign3_SIM_GAM_LAM.log 2>&1 &
PID3=$!
echo "  GPU 3: sign=3 +SIM+GAM+LAM (USE_A=True, USE_B=True, stage=together_CLS_Patch)"
echo "         PID: ${PID3}"
echo "         输出: (..., loss_area, patch_loss)"
echo ""

# 保存 PID 到文件，方便后续管理
echo "${PID0}" > ${EXP_DIR}/pid_gpu0.txt
echo "${PID1}" > ${EXP_DIR}/pid_gpu1.txt
echo "${PID2}" > ${EXP_DIR}/pid_gpu2.txt
echo "${PID3}" > ${EXP_DIR}/pid_gpu3.txt

echo "=============================================================================="
echo "所有实验已启动!"
echo "=============================================================================="
echo ""
echo "监控命令:"
echo "  查看所有日志:  tail -f ${EXP_DIR}/*.log"
echo "  查看 GPU 0:    tail -f ${EXP_DIR}/sign1_baseline.log"
echo "  查看 GPU 1:    tail -f ${EXP_DIR}/sign2_SIM.log"
echo "  查看 GPU 2:    tail -f ${EXP_DIR}/sign3_SIM_GAM.log"
echo "  查看 GPU 3:    tail -f ${EXP_DIR}/sign3_SIM_GAM_LAM.log"
echo ""
echo "GPU 使用情况:    watch -n 1 nvidia-smi"
echo ""
echo "停止所有实验:"
echo "  kill ${PID0} ${PID1} ${PID2} ${PID3}"
echo "  或: kill \$(cat ${EXP_DIR}/pid_gpu*.txt)"
echo "=============================================================================="
