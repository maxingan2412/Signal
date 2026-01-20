"""
Token Selection 注意力分数可视化

使用 Signal 模型 SIM 模块中的 Token Selection 注意力分数生成热力图
这是 Signal 独有的可视化方式，能体现模型选择 token 的逻辑
"""
import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math

from config import cfg
from data import make_dataloader
from modeling import make_frame
from utils.logger import setup_logger


def compute_token_attention(model, img_batch, device):
    """
    计算 Token Selection 的注意力分数

    返回:
        attention_maps: dict, 每个模态的注意力图 [B, H, W]
    """
    model.eval()

    with torch.no_grad():
        RGB = img_batch['RGB']
        NI = img_batch['NI']
        TI = img_batch['TI']
        cam_label = img_batch.get('cam_label', None)
        view_label = img_batch.get('view_label', None)

        # 1. 通过 ViT backbone 提取特征
        rgb_patch, RGB_global = model.clip_vision_encoder(RGB, cam_label=cam_label, view_label=view_label)
        ni_patch, NI_global = model.clip_vision_encoder(NI, cam_label=cam_label, view_label=view_label)
        ti_patch, TI_global = model.clip_vision_encoder(TI, cam_label=cam_label, view_label=view_label)
        # rgb_patch: [B, 128, 512], RGB_global: [B, 512]

        dim = RGB_global.size(-1)  # 512

        # 2. 计算模态内注意力分数 (intra-modal)
        # 公式: score = softmax(CLS · patches^T / √dim)
        rgb_scores = F.softmax(
            (torch.bmm(RGB_global.unsqueeze(1), rgb_patch.transpose(1, 2)) / math.sqrt(dim)).squeeze(1),
            dim=1
        )  # [B, 128]
        ni_scores = F.softmax(
            (torch.bmm(NI_global.unsqueeze(1), ni_patch.transpose(1, 2)) / math.sqrt(dim)).squeeze(1),
            dim=1
        )
        ti_scores = F.softmax(
            (torch.bmm(TI_global.unsqueeze(1), ti_patch.transpose(1, 2)) / math.sqrt(dim)).squeeze(1),
            dim=1
        )

        # 3. Reshape 成空间维度 [B, 16, 8]
        B = RGB.size(0)
        H, W = 16, 8  # ViT patch grid

        rgb_attn = rgb_scores.view(B, H, W)
        ni_attn = ni_scores.view(B, H, W)
        ti_attn = ti_scores.view(B, H, W)

        return {
            'RGB': rgb_attn.cpu().numpy(),
            'NI': ni_attn.cpu().numpy(),
            'TI': ti_attn.cpu().numpy()
        }


def visualize_attention(attn_map, original_img, output_path):
    """
    将注意力图叠加到原始图像上

    Args:
        attn_map: [H, W] 注意力图
        original_img: 原始图像 (numpy array)
        output_path: 保存路径
    """
    # 归一化到 [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # 上采样到原图大小
    attn_map_resized = cv2.resize(attn_map, (original_img.shape[1], original_img.shape[0]))

    # 转换为热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # 叠加
    if original_img.max() > 1:
        original_img = original_img / 255.0

    overlay = 0.5 * heatmap + 0.5 * original_img
    overlay = np.clip(overlay, 0, 1)

    # 保存
    cv2.imwrite(output_path, np.uint8(255 * overlay))


def process_batch(model, img_batch, imgpath, cfg, output_base, device):
    """处理一个 batch 的图像"""
    attention_maps = compute_token_attention(model, img_batch, device)

    batch_size = img_batch['RGB'].size(0)

    for modality in ['RGB', 'NI', 'TI']:
        attn = attention_maps[modality]

        for i in range(batch_size):
            img_name = imgpath[i]

            # 加载原始图像
            if cfg.DATASETS.NAMES == 'RGBNT201':
                img_path = f'../RGBNT201/test/{modality}/{img_name}'
            elif cfg.DATASETS.NAMES == 'RGBNT100':
                img_path = f'../RGBNT100/rgbir/query/{img_name}'
            else:
                img_path = img_name

            img = cv2.imread(img_path, 1)
            if img is None:
                print(f"Warning: Cannot read {img_path}")
                continue
            img = cv2.resize(img, (128, 256))

            # 保存
            output_dir = f'{output_base}/{cfg.DATASETS.NAMES}/{modality}'
            os.makedirs(output_dir, exist_ok=True)
            save_name = img_name.replace('.jpg', '_tokensel.jpg')
            output_path = f'{output_dir}/{save_name}'

            visualize_attention(attn[i], img, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Selection Attention Visualization")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    parser.add_argument("--num_images", default=500, type=int, help="Number of images to visualize")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size (no gradient, can be larger)")
    parser.add_argument("--output_dir", default="tokensel_vis", type=str, help="Output directory")
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("TokenSel", cfg.OUTPUT_DIR, if_train=False)
    logger.info(f"Generating Token Selection visualization for {args.num_images} images")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    model.to(device)

    print(f"Model loaded. Generating Token Selection attention maps...")

    total_processed = 0
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        full_batch = img['RGB'].size(0)

        for start_idx in range(0, full_batch, args.batch_size):
            end_idx = min(start_idx + args.batch_size, full_batch)
            actual_batch = end_idx - start_idx

            img_batch = {
                'RGB': img['RGB'][start_idx:end_idx].to(device),
                'NI': img['NI'][start_idx:end_idx].to(device),
                'TI': img['TI'][start_idx:end_idx].to(device),
                'cam_label': camids[start_idx:end_idx].to(device),
                'view_label': target_view[start_idx:end_idx].to(device)
            }
            imgpath_batch = imgpath[start_idx:end_idx]

            process_batch(model, img_batch, imgpath_batch, cfg, args.output_dir, device)

            total_processed += actual_batch
            print(f"\rProcessed {total_processed}/{args.num_images} images", end="", flush=True)

            if total_processed >= args.num_images:
                break

        if total_processed >= args.num_images:
            break

    print(f"\nToken Selection visualization completed! Results: {args.output_dir}/{cfg.DATASETS.NAMES}/")
