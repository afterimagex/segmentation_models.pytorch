import os
import torch
import shutil
import numpy as np

from PIL import Image
from pathlib import Path


class QrcodeDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = Path(root)
        self.mode = mode
        self.transform = transform

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

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        sample = self._preprocess_sample(sample)

        return sample

    def _preprocess_sample(self, sample):
        # resize images
        # image = np.array(Image.fromarray(sample["image"]).resize((384, 384), Image.BILINEAR))
        # mask = np.array(Image.fromarray(sample["mask"]).resize((384, 384), Image.NEAREST))
        # trimap = np.array(Image.fromarray(sample["trimap"]).resize((384, 384), Image.NEAREST))
        image = Image.fromarray(sample["image"])
        mask = np.array(Image.fromarray(sample["mask"]))
        trimap = np.array(Image.fromarray(sample["trimap"]))

        # convert to other format HWC -> CHW
        # sample["image"] = np.moveaxis(image, -1, 0)
        sample["image"] = np.expand_dims(image, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

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
            filenames = [x for i, x in enumerate(filenames) if i % 500 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 500 == 0]
        elif self.mode == "test":
            filenames = [x for i, x in enumerate(filenames) if i % 500 == 0]
        return filenames
