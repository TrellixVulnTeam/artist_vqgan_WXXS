import bisect
import copy
import random

import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from taming.data.augmentations import AugmentPipe


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, augmentations=None, disc_augmentations=None, labels=None, *args):
        self.size = size
        self.disc_data = False

        augmentations = copy.deepcopy(augmentations)
        if augmentations is None:
            augmentations = dict()
        self.aug_p = augmentations['p'] if 'p' in augmentations else 1.

        basic_operations = dict()
        basic_operations['rescale'] = augmentations['rescale'] if 'rescale' in augmentations else 1.
        basic_operations['crop'] = augmentations['crop'] if 'crop' in augmentations else 'center'
        augmentations.pop('rescale', None)
        augmentations.pop('crop', None)

        disc_augmentations = copy.deepcopy(disc_augmentations)
        if disc_augmentations is None:
            disc_augmentations = {}
        self.disc_aug_p = disc_augmentations['p'] if 'p' in disc_augmentations else 0.

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.preprocessor_basic = AugmentPipe(self.size, basic_operations)
        self.preprocessor_aug = AugmentPipe(self.size, augmentations)
        self.preprocessor_disc_aug = AugmentPipe(self.size, disc_augmentations)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        image = self.preprocessor_basic(image)
        if random.random() < self.aug_p:
            image = self.preprocessor_aug(image)

        if self.disc_data and self.disc_aug_p > 0:
            disc_image = copy.deepcopy(image)
            if random.random() < self.disc_aug_p:
                disc_image = self.preprocessor_disc_aug(disc_image)
                disc_image = (disc_image/127.5 - 1.0).astype(np.float32)
        else:
            disc_image = image

        image = (image/127.5 - 1.0).astype(np.float32)
        return image, disc_image

    def __getitem__(self, i):
        example = dict()
        example["image"], example["disc_image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor_basic(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
