import os
import random
import warnings

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from taming.data.base import ImagePaths

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def test_images(root, images):
    passed_images = list()
    for fname in tqdm(images):
        with warnings.catch_warnings(record=True) as caught_warnings:
            image = np.array(Image.open(os.path.join(root, fname)))
            if len(caught_warnings) > 0:
                continue
        if image.ndim == 3 and image.shape[-1] not in [1, 3, 4]:
            continue
        passed_images.append(fname)

    return passed_images


def get_all_images(root):
    train_file = os.path.join(root, 'train.npy')
    val_file = os.path.join(root, 'val.npy')

    if not os.path.isfile(train_file) or not os.path.isfile(val_file):
        images = list()
        for category in [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]:
            category_dir = os.path.join(root, category)
            images.extend([os.path.join(category, fname) for fname in os.listdir(category_dir)
                           if os.path.splitext(fname)[1].lower() in ['.jpg', '.jpeg', '.png']])
        passed_images = test_images(root, images)

        random.shuffle(passed_images)
        num_train_images = int(len(passed_images) * 0.9)
        images_train = np.array(passed_images[:num_train_images])
        images_val = np.array(passed_images[num_train_images:])

        np.save(train_file, images_train)
        np.save(val_file, images_val)

    else:
        images_train = np.load(train_file)
        images_val = np.load(val_file)

    return {'train': images_train, 'val': images_val}


class WikiArtBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class WikiArtTrain(WikiArtBase):
    def __init__(self, size, base=None, ae_augmentations=None, disc_augmentations=None, *args):
        super().__init__()
        root = "/data/datasets/art/wiki-art/"
        relpaths = get_all_images(root)['train']
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = \
            ImagePaths(paths=paths, size=size, base=base,
                       ae_augmentations=ae_augmentations, disc_augmentations=disc_augmentations)

        print(f'total {len(self.data)} training data.')


class WikiArtValidation(WikiArtBase):
    def __init__(self, size, base=None, *args):
        super().__init__()
        root = "/data/datasets/art/wiki-art/"
        relpaths = get_all_images(root)['val']
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, base=base)

        print(f'total {len(self.data)} validation data.')


class NatgeoTesing(WikiArtBase):
    def __init__(self, size, base=None, *args):
        super(NatgeoTesing, self).__init__()
        root = '/data/datasets/photography/natgeo/'
        paths = [os.path.join(root, fname) for fname in os.listdir(root)
                 if os.path.splitext(fname)[1].lower() in ['.jpg', '.jpeg', '.png']]
        self.data = ImagePaths(paths=sorted(paths), size=size, base=base)

        print(f'total {len(self.data)} testing data.')
