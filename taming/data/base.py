import bisect
import copy
from multiprocessing import Array
import ctypes

import numpy as np
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
    def __init__(self, paths, size=None, base=None, augmentations=None, disc_augmentations=None, labels=None, *args):
        self.size = size

        if base is None:
            base = dict()
        base['p'] = 1.
        if augmentations is None:
            augmentations = dict()
        if disc_augmentations is None:
            disc_augmentations = {}

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.preprocessor_basic = AugmentPipe(self.size, base)
        self.preprocessor_aug = AugmentPipe(self.size, augmentations)
        self.preprocessor_disc_aug = AugmentPipe(self.size, disc_augmentations)

        self._prepare_disc_data = np.ctypeslib.as_array(Array(ctypes.c_bool, [False]).get_obj())
        self._adaptive_disc_p = \
            np.ctypeslib.as_array(Array(ctypes.c_float, [disc_augmentations.get('p', 0.)]).get_obj())

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        image = self.preprocessor_basic(image)
        image_g = self.preprocessor_aug(image)
        if self.prepare_disc_data():
            image_d = self.preprocessor_disc_aug(image, self.disc_aug_p())
        else:
            image_d = copy.deepcopy(image_g)
        image_g = (image_g/127.5 - 1.0).astype(np.float32)
        image_d = (image_d/127.5 - 1.0).astype(np.float32)

        return image_g, image_d

    def __getitem__(self, i):
        example = dict()
        example["image"], example["disc_image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    def prepare_disc_data(self, status=None):
        if status is not None:
            self._prepare_disc_data[0] = status
        return self._prepare_disc_data[0]

    def adjust_disc_aug_p(self, adjust):
        self._adaptive_disc_p[0] = np.clip(self._adaptive_disc_p[0] + adjust, 0., 1.)

    def disc_aug_p(self):
        return self._adaptive_disc_p[0]


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor_basic(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
