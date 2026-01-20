"""
Grad-CAM 可视化脚本

使用 ViT 最后一层 Transformer Block (resblocks[-1]) 生成热力图
"""
import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import cfg
from data import make_dataloader
from modeling import make_frame
from utils.logger import setup_logger


class Newdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = [1, 2, 3, 4]

    def to(self, device):
        for key in self.keys():
            self[key] = self[key].to(device)
        return self

    def size(self, k):
        data = self['RGB']
        width, height = data.size(-1), data.size(-2)
        return width if k == -1 else height


class ModelWrapper(torch.nn.Module):
    """包装 Signal 模型以适配 pytorch_grad_cam 接口"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.clip_vision_encoder = model.clip_vision_encoder

    def forward(self, x):
        cam_label = x.get('cam_label', None)
        view_label = x.get('view_label', None)
        output = self.model(x, cam_label=cam_label, view_label=view_label, training=False)
        return output


def reshape_transform(tensor, height=16, width=8):
    """
    将 ViT resblock 输出转换为 [B, D, H, W] 用于可视化
    resblock 输出格式: [L, B, D] (LND format)，其中 L = 129 (1 CLS + 128 patches)
    """
    # 检测输入格式
    if tensor.dim() == 3:
        if tensor.size(0) == 129:  # LND format: [129, B, 768]
            tensor = tensor.permute(1, 0, 2)  # LND -> NLD: [B, 129, 768]

    # 现在是 [B, L, D] 格式，移除 CLS token
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)  # [B, D, H, W]


def show_cam(index, imgpath, grayscale_cam, modality, cfg, n_iter, output_base):
    index = int(index)
    img_path = imgpath[index]

    if cfg.DATASETS.NAMES == 'RGBNT201':
        full_path = f'../RGBNT201/test/{modality}/{img_path}'
    elif cfg.DATASETS.NAMES == 'RGBNT100':
        full_path = f'../RGBNT100/rgbir/query/{img_path}'
    else:
        full_path = img_path

    grayscale_cam = grayscale_cam[index]

    if cfg.DATASETS.NAMES == 'RGBNT100':
        img = Image.open(full_path).convert('RGB')
        if modality == "RGB":
            cropped_image = img.crop((0, 0, 256, 128))
        elif modality == "NI":
            cropped_image = img.crop((256, 0, 512, 128))
        else:
            cropped_image = img.crop((512, 0, 768, 128))
        rgb_image = np.float32(cropped_image) / 255
    else:
        img = cv2.imread(full_path, 1)
        if img is None:
            print(f"Warning: Cannot read {full_path}")
            return
        rgb_image = cv2.resize(img, (128, 256))
        rgb_image = np.float32(rgb_image) / 255

    visualization = show_cam_on_image(rgb_image, grayscale_cam)
    output_dir = f'{output_base}/{cfg.DATASETS.NAMES}/{modality}'
    os.makedirs(output_dir, exist_ok=True)

    # 使用原始文件名
    save_name = img_path.replace('.jpg', '_gradcam.jpg')
    save_path = f'{output_dir}/{save_name}'
    cv2.imwrite(save_path, visualization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization (resblocks[-1])")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    parser.add_argument("--num_images", default=500, type=int, help="Number of images to visualize")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for processing")
    parser.add_argument("--output_dir", default="gradcam_vis", type=str, help="Output directory")
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("GradCAM", cfg.OUTPUT_DIR, if_train=False)
    logger.info(f"Generating Grad-CAM for {args.num_images} images")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    model.to(device)

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    # 使用 ViT 最后一层 Transformer Block
    target_layers = [wrapped_model.clip_vision_encoder.base.transformer.resblocks[-1]]
    print(f"Target layer: transformer.resblocks[-1] (last Transformer block)")

    total_processed = 0
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        # 使用完整 batch（但限制显存使用，分批处理）
        full_batch = img['RGB'].size(0)

        for start_idx in range(0, full_batch, args.batch_size):
            end_idx = min(start_idx + args.batch_size, full_batch)
            actual_batch = end_idx - start_idx

            img_batch = Newdict({
                'RGB': img['RGB'][start_idx:end_idx].to(device),
                'NI': img['NI'][start_idx:end_idx].to(device),
                'TI': img['TI'][start_idx:end_idx].to(device),
                'cam_label': camids[start_idx:end_idx].to(device),
                'view_label': target_view[start_idx:end_idx].to(device)
            })
            imgpath_batch = imgpath[start_idx:end_idx]

            cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
            grayscale_cam = cam(input_tensor=img_batch, targets=None, eigen_smooth=False, aug_smooth=False)

            for modality in ["RGB", "NI", "TI"]:
                for i in range(actual_batch):
                    show_cam(i, imgpath_batch, grayscale_cam, modality, cfg, n_iter, args.output_dir)

            total_processed += actual_batch
            print(f"\rProcessed {total_processed}/{args.num_images} images", end="", flush=True)

            if total_processed >= args.num_images:
                break

        if total_processed >= args.num_images:
            break

    print(f"\nGrad-CAM completed! Results: {args.output_dir}/{cfg.DATASETS.NAMES}/")
