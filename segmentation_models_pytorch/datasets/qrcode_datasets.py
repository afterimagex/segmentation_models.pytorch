import os
import cv2
import torch
import shutil
import numpy as np

from PIL import Image
from pathlib import Path
import albumentations as albu


class QrcodeDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", augmentation=None, preprocessing=None):

        assert mode in {"train", "valid", "test"}

        self.root = Path(root)
        self.mode = mode
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.images_directory = self.root / "defective"
        self.masks_directory = self.root / "mask"

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = self.images_directory / filename
        mask_path = self.masks_directory / filename

        image = np.array(Image.open(image_path).convert("L"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)
        
        sample = dict(image=image, mask=mask)

        if self.augmentation:
            sample = self.augmentation(**sample)

        # (H, W) -> (H, W, C)
        sample["image"] = np.expand_dims(sample["image"], 2)
        sample["mask"] = np.expand_dims(sample["mask"], 2)
        
        if self.preprocessing:
            sample = self.preprocessing(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask = np.where(mask == 127.0, 1.0, 0.0)
        return mask

    def _read_split(self):
        filenames = []
        for filepath in self.images_directory.rglob('*.png'):
            if (self.masks_directory / filepath.name).exists():
                filenames.append(str(filepath.name))
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        elif self.mode == "test":
            filenames = [x for i, x in enumerate(filenames) if i % 11 == 0]
        return filenames


class QRAugmentation(object):

    @staticmethod
    def get_training_augmentation():
        return albu.Compose([
            albu.HorizontalFlip(p=0.5),

            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.01, p=1, 
                    border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST, value=255, mask_value=0),

            albu.OneOf([
                albu.Compose([
                    albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, 
                        border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                    albu.RandomCrop(height=320, width=320, always_apply=True),
                ]),
                albu.Lambda(image=QRAugmentation.resize_and_pad, mask=QRAugmentation.resize_and_pad),
                albu.Lambda(image=QRAugmentation.letterbox, mask=QRAugmentation.letterbox),
            ], p=1.0),

            albu.IAAAdditiveGaussianNoise(p=0.2),
            albu.IAAPerspective(p=0.5),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    # hue_shift and sat_shift are not applicable to grayscale image. Set them to 0 or use RGB image
                    albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, p=1), 
                ],
                p=0.9,
            ),
        ])

    @staticmethod
    def letterbox(img, new_shape=(320, 320), color=(255, 255, 255),
              auto=True, scale_fill=False, scale_up=True, **kwargs):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img
    
    @staticmethod
    def resize_and_pad(img, dsize=(320, 320), channel=1, border_color=255, **kwargs):
        '''
        :param img: (H, W, C)
        :param dsize: (W, H) 目标大小
        :return:
        '''
        ih, iw = img.shape[:2]
        dw, dh = dsize

        # 计算最优目标 (nw, nh)
        max_wh_ratio = max(float(iw) / ih, float(dw) / dh) # 获取最大宽高比
        nh = dh
        nw = max_wh_ratio * dh
        nw = int(int(nw / 32.0 + 0.5) * 32) # 32倍数四舍五入

        if float(iw) / ih > float(nw) / nh:
            # 图宽了
            ratio = 1.0 * nw / iw
        else:
            # 图高了
            ratio = 1.0 * nh / ih
        
        # 保持宽高比缩放
        resized = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        new_image = np.zeros((nh, nw), dtype=np.uint8) + border_color
        new_image[:resized.shape[0], :resized.shape[1]] = resized
        return new_image

    @staticmethod 
    def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        return albu.Compose([
            albu.OneOf([
                albu.Lambda(image=QRAugmentation.resize_and_pad, mask=QRAugmentation.resize_and_pad),
                albu.Lambda(image=QRAugmentation.letterbox, mask=QRAugmentation.letterbox),
            ], p=1.0),
        ])

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    @staticmethod
    def get_preprocessing(preprocessing_fn=None):
        """Construct preprocessing transform
    
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose
        
        """
        return albu.Compose([
            # albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=QRAugmentation.to_tensor, mask=QRAugmentation.to_tensor),
        ])