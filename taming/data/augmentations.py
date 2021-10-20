import albumentations


class AugmentPipe:
    def __init__(self, size=None, augment_types=None):
        self.size = size

        if augment_types is None or self.size is None or self.size <= 0:
            self.processor = lambda **kwargs: kwargs
        else:
            transforms = list()

            if augment_types.get('rescale') is not None:
                assert augment_types['rescale'] >= 1.
                rescale_size = augment_types['rescale'] * self.size
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

            if augment_types.get('hflip'):
                self.hflip = albumentations.HorizontalFlip(p=0.5)
                transforms.append(self.hflip)
            if augment_types.get('vflip'):
                self.vflip = albumentations.VerticalFlip(p=0.5)
                transforms.append(self.vflip)

            if augment_types.get('bright_contrast') is not None:
                p = augment_types['bright_contrast']['p']
                limit = tuple(augment_types['bright_contrast']['limit'])
                assert 0 <= p <= 1
                self.bright_contrast = \
                    albumentations.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=limit, p=p)
                transforms.append(self.bright_contrast)

            self.processor = albumentations.Compose(transforms)

    def __call__(self, image):
        return self.processor(image=image)['image']
