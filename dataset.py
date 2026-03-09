import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

class SCIDataset(Dataset):
    # 注意：现在连 lq_dir 都不需要传了！只需要高清图目录
    def __init__(self, hq_dir, patch_size=256, multiplier=100):
        self.hq_dir = hq_dir
        self.patch_size = patch_size
        self.image_names = [f for f in os.listdir(hq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        # 倍率放大，弥补40张图数据量不足的问题
        self.multiplier = multiplier

    def __len__(self):
        return len(self.image_names) * self.multiplier

    def extract_edge_mask(self, hq_cv_rgb):
        """基于 Scharr 算子的纹理自适应 Mask 提取 (直接处理 numpy 数组)"""
        gray = cv2.cvtColor(hq_cv_rgb, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, mask = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (mask > 0).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, idx):
        # 确保索引在 0-39 之间循环
        actual_idx = idx % len(self.image_names)
        img_name = self.image_names[actual_idx]
        
        # 1. 仅仅读取高质量原图
        hq_pil = Image.open(os.path.join(self.hq_dir, img_name)).convert('RGB')

        # 2. 几何增强 (只对 HQ 做)
        i, j, h, w = transforms.RandomCrop.get_params(hq_pil, output_size=(self.patch_size, self.patch_size))
        hq_crop = TF.crop(hq_pil, i, j, h, w)

        if random.random() > 0.5:
            hq_crop = TF.hflip(hq_crop)
        if random.random() > 0.5:
            hq_crop = TF.vflip(hq_crop)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hq_crop = TF.rotate(hq_crop, angle)

        # ====== 核心魔法：在这里使用你的 OpenCV 代码实时生成 LQ ======
        # PIL (RGB) 转 numpy 数组
        hq_cv = np.array(hq_crop)
        # 转成 OpenCV 默认的 BGR 格式
        hq_bgr = cv2.cvtColor(hq_cv, cv2.COLOR_RGB2BGR)

        # 随机生成压缩质量 (10 到 50 之间)
        random_q = random.randint(10, 50)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random_q]
        success, encimg = cv2.imencode('.jpg', hq_bgr, encode_param)

        if success:
            # 内存解码，此时带上了真实的 JPEG 瑕疵
            decoded_bgr = cv2.imdecode(encimg, 1)
            # 再转回 RGB
            lq_cv = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
            lq_crop = Image.fromarray(lq_cv)
        else:
            # 万一压缩失败，用原图兜底
            lq_crop = hq_crop
        # =========================================================

        # 4. 提取 Mask (直接传增强后的 HQ numpy 数组)
        gt_mask = self.extract_edge_mask(hq_cv)

        # 5. 转为 Tensor 输出给神经网络
        hq_tensor = TF.to_tensor(hq_crop)
        lq_tensor = TF.to_tensor(lq_crop)

        return lq_tensor, hq_tensor, gt_mask