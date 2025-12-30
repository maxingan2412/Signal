#!/bin/bash
# ============================================================================
# Signal 完整消融实验 Pipeline
#
# 4 种模型配置 × 3 个数据集 = 12 个实验
#   - baseline:         无 SIM, 无 GAM/LAM
#   - baseline+SIM:     有 SIM, 无 GAM/LAM
#   - baseline+SIM+GAM: 有 SIM, 有 GAM (stage=CLS)
#   - baseline+SIM+GAM+LAM: 有 SIM, 有 GAM+LAM (stage=together_CLS_Patch)
#
# 数据集: RGBNT201, MSVR310, RGBNT100
# GPU: 0-3 并行，每个数据集一组，顺序执行
#
# 用法:
#   bash run_ablation_signal.sh [实验标识]
#
# 示例:
#   bash run_ablation_signal.sh v1
#   bash run_ablation_signal.sh  # 默认标识为 "exp"
# ============================================================================

EXP_TAG="${1:-exp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 数据集列表
DATASETS=("RGBNT201" "MSVR310" "RGBNT100")

# 输出根目录（Signal 的上一级目录）
OUTPUT_ROOT="${SCRIPT_DIR}/.."

echo "=============================================================================="
echo "Signal 完整消融实验 Pipeline"
echo "=============================================================================="
echo "实验标识: ${EXP_TAG}"
echo "开始时间: $(date)"
echo "输出根目录: ${OUTPUT_ROOT}"
echo ""
echo "实验配置 (每个数据集 4 个实验):"
echo "  1. baseline:             USE_A=False, USE_B=False"
echo "  2. baseline+SIM:         USE_A=True,  USE_B=False"
echo "  3. baseline+SIM+GAM:     USE_A=True,  USE_B=True, stage=CLS"
echo "  4. baseline+SIM+GAM+LAM: USE_A=True,  USE_B=True, stage=together_CLS_Patch"
echo ""
echo "数据集: ${DATASETS[*]}"
echo "=============================================================================="

# ============================================================================
# 工具函数
# ============================================================================

# 等待 PID 完成
wait_for_pids() {
    local log_dir=$1
    local pids=()

    for pid_file in ${log_dir}/pid_gpu*.txt; do
        if [ -f "$pid_file" ]; then
            pids+=($(cat "$pid_file"))
        fi
    done

    if [ ${#pids[@]} -eq 0 ]; then
        echo "  警告: 未找到 PID 文件"
        return
    fi

    echo "  等待 ${#pids[@]} 个进程完成: ${pids[*]}"

    local running=true
    while $running; do
        running=false
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                running=true
                break
            fi
        done
        if $running; then
            sleep 60  # 每分钟检查一次
        fi
    done

    echo "  所有进程已完成!"
}

# 打印实验结果摘要
print_results() {
    local log_dir=$1
    echo ""
    echo "  结果摘要:"
    for log in ${log_dir}/*.log; do
        if [ -f "$log" ]; then
            local name=$(basename "$log" .log)
            echo "    ${name}:"
            grep -E "(Best mAP|Best Rank-1)" "$log" 2>/dev/null | tail -2 | sed 's/^/      /'
        fi
    done
}

# 运行单个数据集的 4 个消融实验
run_dataset_ablation() {
    local dataset=$1
    local config_file="configs/${dataset}/Signal.yml"

    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        echo "  错误: 配置文件不存在 ${config_file}"
        return 1
    fi

    # 创建该数据集的输出目录（在 Signal 上一级）
    local exp_dir="${OUTPUT_ROOT}/Signal_${dataset}_${EXP_TAG}_${TIMESTAMP}"
    local log_dir="${SCRIPT_DIR}/logs/${dataset}_${EXP_TAG}_${TIMESTAMP}"
    mkdir -p "${exp_dir}"
    mkdir -p "${log_dir}"

    echo ""
    echo "  配置文件: ${config_file}"
    echo "  模型输出: ${exp_dir}"
    echo "  日志目录: ${log_dir}"
    echo ""

    # GPU 0: baseline (无 SIM, 无 GAM/LAM)
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --config_file ${config_file} \
        MODEL.USE_A False \
        MODEL.USE_B False \
        OUTPUT_DIR "${exp_dir}/baseline" \
        > ${log_dir}/baseline.log 2>&1 &
    local PID0=$!
    echo "${PID0}" > ${log_dir}/pid_gpu0.txt
    echo "    GPU 0: baseline (PID: ${PID0})"

    # GPU 1: baseline+SIM (有 SIM, 无 GAM/LAM)
    CUDA_VISIBLE_DEVICES=1 nohup python train.py --config_file ${config_file} \
        MODEL.USE_A True \
        MODEL.USE_B False \
        OUTPUT_DIR "${exp_dir}/baseline+SIM" \
        > ${log_dir}/baseline+SIM.log 2>&1 &
    local PID1=$!
    echo "${PID1}" > ${log_dir}/pid_gpu1.txt
    echo "    GPU 1: baseline+SIM (PID: ${PID1})"

    # GPU 2: baseline+SIM+GAM (有 SIM, 有 GAM, stage=CLS)
    CUDA_VISIBLE_DEVICES=2 nohup python train.py --config_file ${config_file} \
        MODEL.USE_A True \
        MODEL.USE_B True \
        MODEL.stageName "CLS" \
        OUTPUT_DIR "${exp_dir}/baseline+SIM+GAM" \
        > ${log_dir}/baseline+SIM+GAM.log 2>&1 &
    local PID2=$!
    echo "${PID2}" > ${log_dir}/pid_gpu2.txt
    echo "    GPU 2: baseline+SIM+GAM (PID: ${PID2})"

    # GPU 3: baseline+SIM+GAM+LAM (有 SIM, 有 GAM+LAM, stage=together_CLS_Patch)
    CUDA_VISIBLE_DEVICES=3 nohup python train.py --config_file ${config_file} \
        MODEL.USE_A True \
        MODEL.USE_B True \
        MODEL.stageName "together_CLS_Patch" \
        OUTPUT_DIR "${exp_dir}/baseline+SIM+GAM+LAM" \
        > ${log_dir}/baseline+SIM+GAM+LAM.log 2>&1 &
    local PID3=$!
    echo "${PID3}" > ${log_dir}/pid_gpu3.txt
    echo "    GPU 3: baseline+SIM+GAM+LAM (PID: ${PID3})"

    echo ""
    echo "  监控命令: tail -f ${log_dir}/*.log"

    # 等待所有进程完成
    wait_for_pids "${log_dir}"
    print_results "${log_dir}"

    return 0
}

# ============================================================================
# 主流程：按数据集顺序执行
# ============================================================================

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"

    echo ""
    echo "=============================================================================="
    echo "[${dataset}] ($((i+1))/${#DATASETS[@]}) 开始执行"
    echo "时间: $(date)"
    echo "=============================================================================="

    run_dataset_ablation "${dataset}"

    # 如果不是最后一个数据集，休息一下
    if [ $i -lt $((${#DATASETS[@]} - 1)) ]; then
        echo ""
        echo "[$(date +%H:%M:%S)] 休息 30 秒后启动下一组..."
        sleep 30
    fi
done

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "=============================================================================="
echo "所有实验完成!"
echo "=============================================================================="
echo "结束时间: $(date)"
echo ""
echo "实验结果目录:"
for dataset in "${DATASETS[@]}"; do
    echo "  ${dataset}: ${OUTPUT_ROOT}/Signal_${dataset}_${EXP_TAG}_${TIMESTAMP}/"
done
echo ""
echo "日志目录:"
for dataset in "${DATASETS[@]}"; do
    echo "  ${dataset}: ${SCRIPT_DIR}/logs/${dataset}_${EXP_TAG}_${TIMESTAMP}/"
done
echo "=============================================================================="
