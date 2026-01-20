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
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def show_cam(index, imgpath, grayscale_cam, modality, cfg, n_iter):
    index = int(index)
    img_path = imgpath[index]
    print(f"Processing {modality} {index}: {img_path}")

    if cfg.DATASETS.NAMES == 'RGBNT201':
        img_path = f'../RGBNT201/test/{modality}/{img_path}'
    elif cfg.DATASETS.NAMES == 'RGBNT100':
        img_path = f'../RGBNT100/rgbir/query/{img_path}'

    grayscale_cam = grayscale_cam[index]
    if cfg.DATASETS.NAMES == 'RGBNT100':
        img = Image.open(img_path).convert('RGB')
        if modality == "RGB":
            cropped_image = img.crop((0, 0, 256, 128))
        elif modality == "NI":
            cropped_image = img.crop((256, 0, 512, 128))
        else:
            cropped_image = img.crop((512, 0, 768, 128))
        rgb_image = np.float32(cropped_image) / 255
    else:
        img = cv2.imread(img_path, 1)
        rgb_image = cv2.resize(img, (128, 256))
        rgb_image = np.float32(rgb_image) / 255

    visualization = show_cam_on_image(rgb_image, grayscale_cam)
    output_dir = f'gradcam_vis/{cfg.DATASETS.NAMES}/{modality}'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/{n_iter * cfg.TEST.IMS_PER_BATCH + index}.jpg'
    cv2.imwrite(save_path, visualization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    parser.add_argument("--num_images", default=10, type=int, help="Number of images to visualize")
    parser.add_argument("opts", help="Modify config options via command line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("GradCAM", output_dir, if_train=False)
    logger.info(args)

    if args.config_file:
        logger.info(f"Loaded configuration file {args.config_file}")
    logger.info(f"Running with config:\n{cfg}")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    model.to(device)

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    target_layers = [wrapped_model.clip_vision_encoder.base]

    total_processed = 0
    batch_size = 8  # 减小 batch size 避免 OOM
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        # 只取前 batch_size 个样本
        actual_batch = min(batch_size, img['RGB'].size(0))
        img = Newdict({'RGB': img['RGB'][:actual_batch].to(device),
                       'NI': img['NI'][:actual_batch].to(device),
                       'TI': img['TI'][:actual_batch].to(device),
                       'cam_label': camids[:actual_batch].to(device),
                       'view_label': target_view[:actual_batch].to(device)})
        imgpath = imgpath[:actual_batch]

        cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
        grayscale_cam = cam(input_tensor=img, targets=None, eigen_smooth=False, aug_smooth=False)

        for modality in ["RGB", "NI", "TI"]:
            for i in range(actual_batch):
                show_cam(i, imgpath, grayscale_cam, modality, cfg, n_iter)

        total_processed += actual_batch
        print(f"Processed {total_processed} images")

        if total_processed >= args.num_images:
            break

    print(f"Grad-CAM visualization completed! Results saved in gradcam_vis/{cfg.DATASETS.NAMES}/")
