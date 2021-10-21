import ctypes
import numbers
from multiprocessing import Array

import numpy as np
import albumentations
import cv2


class AugmentPipe:
    def __init__(self, size=None, augment_types=None):
        self.size = size

        if augment_types is None or self.size is None or self.size <= 0:
            self.processor = lambda **kwargs: kwargs
        else:
            transforms = list()

            if augment_types.get('rescale') is not None:
                if isinstance(augment_types['rescale'], numbers.Number):
                    rescale_size = int(augment_types['rescale'] * self.size)
                else:
                    rescale_size = [int(s * self.size) for s in augment_types['rescale']]
                self.rescaler = albumentations.SmallestMaxSize(max_size=rescale_size)
                transforms.append(self.rescaler)

            if augment_types.get('crop') is not None:
                crop_type = augment_types['crop']
                assert crop_type in ['center', 'random']
                if crop_type == 'center':
                    self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
                elif crop_type == 'random':
                    self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
                transforms.append(self.cropper)

            hf_cfg = augment_types.get('hflip')
            if hf_cfg is not None:
                self.hflip = albumentations.HorizontalFlip(**hf_cfg)
                transforms.append(self.hflip)
            vf_cfg = augment_types.get('vflip')
            if vf_cfg is not None:
                self.vflip = albumentations.VerticalFlip(**vf_cfg)
                transforms.append(self.vflip)

            f_p_cfg = augment_types.get('fancy_pca')
            if f_p_cfg is not None:
                self.fancy_pca = albumentations.FancyPCA(**f_p_cfg)
                transforms.append(self.fancy_pca)

            c_j_cfg = augment_types.get('color_jitter')
            if c_j_cfg is not None:
                self.color_jitter = \
                    albumentations.ColorJitter(**c_j_cfg)
                transforms.append(self.color_jitter)

            f_cfg = augment_types.get('fog')
            if f_cfg is not None:
                self.fog = \
                    albumentations.RandomFog(**f_cfg)
                transforms.append(self.fog)

            s_s_r_cfg = augment_types.get('shift_scale_rotate')
            if s_s_r_cfg is not None:
                self.shift_scale_rotate = \
                    albumentations.ShiftScaleRotate(**s_s_r_cfg, interpolation=cv2.INTER_LINEAR)
                transforms.append(self.shift_scale_rotate)

            self.processor = albumentations.Compose(transforms)
            self.origin_ps = [trans.p for trans in self.processor]

    def __call__(self, image, adaptive_p=1.):
        if adaptive_p < 1.:
            for i, trans in enumerate(self.processor):
                trans.p = self.origin_ps[i] * adaptive_p
        return self.processor(image=image)['image']
