import os
import sys
import numpy as np
import albumentations as A
from typing import Union, List, Dict
from stainlib import LuminosityStandardizer
from stainlib.augmentation.augmenter import StainAugmentor as StainlibStainAugmentor
from stainlib.augmentation.augmenter import HedColorAugmenter
from albumentations.core.transforms_interface import ImageOnlyTransform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pytorch_models.utils.utils import to_tuple
from he_preprocessing.utils.image import replace_pixels


def get_augmentor(
    patch_size=512,
    enable_augmentation=True,
    enable_stain_augmentation=True,
    replace_background=True,
    constant_pad_value=230,
    split="train",
    additional_targets: Dict[str, str] = None,
):
    assert split in ["train", "test", "val"]
    PATCH_SIZE = patch_size

    transforms = []

    if enable_augmentation and split == "train":
        # Horizontal Flip
        transforms.append(A.HorizontalFlip(p=0.5))

        # Vertical Flip
        transforms.append(A.VerticalFlip(p=0.5))

        # Rotate 90
        transforms.append(A.RandomRotate90(p=0.5))

        # Transpose
        transforms.append(A.Transpose(p=0.5))

        # Brightness and Contrast
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3
            )
        )

        # Sharpen
        transforms.append(A.Sharpen(alpha=(0.0, 0.5), p=0.3))

        # Gaussian Noise
        transforms.append(A.GaussNoise(p=0.3))

        # # Gaussian Blur
        # # TODO: Is this relevant??? We discard blurry patches anyway
        # transforms.append(A.GaussianBlur(p=0.3))

        # Stain augmentation
        if enable_stain_augmentation:
            transforms.append(
                A.OneOf(
                    [
                        # TODO: add this augmentor https://digitalslidearchive.github.io/HistomicsTK/examples/color_normalization_and_augmentation.html#%E2%80%9CSmart%E2%80%9D-color-augmentation
                        # HED Augmentation
                        HEDAugmentor(p=0.25),
                        # HSV Augmentation
                        A.HueSaturationValue(
                            hue_shift_limit=(-10, 10),
                            sat_shift_limit=(-10, 10),
                            val_shift_limit=0,
                            p=0.25,
                        ),
                        # TODO: These two augmentations result in cpu deadlocking.
                        # # Macenko Augmentation
                        # StainAugmentor(
                        #     luminosity=True,
                        #     method="macenko",
                        #     sigma1=0.2,
                        #     sigma2=0.2,
                        #     p=0.25,
                        # ),
                        # # Vahadane Augmentation
                        # StainAugmentor(
                        #     luminosity=True,
                        #     method="vahadane",
                        #     sigma1=0.2,
                        #     sigma2=0.2,
                        #     p=0.25,
                        # ),
                    ],
                    p=0.3,
                )
            )

        # Elastic of Affine (scale, translation, rotation and shear)
        transforms.append(
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                    A.Affine(
                        scale=[0.9, 1.1],
                        translate_percent=0.1,
                        rotate=[-30, 30],
                        shear=[-9, 9],
                        p=0.3,
                    ),
                    # A.Affine(
                    #     scale=[0.8, 1.2],
                    #     translate_percent=0.2,
                    #     rotate=[-90, 90],
                    #     shear=[-0.9, 0.9],
                    #     p=0.3,
                    # ),
                ],
                p=0.3,
            )
        )

    # Ensure image size is PATCH_SIZE and apply some transforms
    transforms.append(
        A.LongestMaxSize(max_size=PATCH_SIZE, p=1.0),
    )

    # Replace black with custom color
    if replace_background:
        transforms.append(
            ReplaceBackgroundColor(
                old_color=0,
                new_color=constant_pad_value,
                rgb=True,
                always_apply=True,
                p=1.0,
            ),
        )

    # Return current split transform
    return A.Compose(transforms, additional_targets=additional_targets)


class ReplaceBackgroundColor(ImageOnlyTransform):
    def __init__(
        self,
        old_color: Union[int, List[int]],
        new_color: Union[int, List[int]],
        rgb: bool = True,
        always_apply=True,
        p=1.0,
    ):
        super(ReplaceBackgroundColor, self).__init__(always_apply, p)
        self.old_color = to_tuple(old_color, shape=3 if rgb else 2)
        self.new_color = to_tuple(new_color, shape=3 if rgb else 2)

    def get_transform_init_args_names(self):
        return "old_color", "new_color"

    def apply(self, img, **params):
        return np.array(
            replace_pixels(img, old_color=self.old_color, new_color=self.new_color)
        ).astype(np.uint8)


class HEDAugmentor(ImageOnlyTransform):
    def __init__(self, h_sigma=0.1, e_sigma=0.1, always_apply=False, p=0.5):
        super(HEDAugmentor, self).__init__(always_apply, p)
        self.h_sigma = h_sigma
        self.e_sigma = e_sigma
        haematoxylin_sigma_range = (
            -h_sigma,
            h_sigma,
        )  # (tuple, None): Adjustment range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.1, 0.1).
        haematoxylin_bias_range = (
            -0.0,
            0.0,
        )  # (tuple, None): Bias range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.2, 0.2).
        eosin_sigma_range = (
            -e_sigma,
            e_sigma,
        )  # (tuple, None): Adjustment range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
        eosin_bias_range = (
            -0.0,
            0.0,
        )  # (tuple, None) Bias range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
        dab_sigma_range = (
            -0.0,
            0.0,
        )  # (tuple, None): Adjustment range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
        dab_bias_range = (
            -0.0,
            0.0,
        )  # (tuple, None): Bias range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
        cutoff_range = (0.05, 0.95)  # (tuple, None) #ignore almost empty patches
        # cutoff_range = (0.0, 1.0)  # (tuple, None):
        self.hed_augmentor = HedColorAugmenter(
            haematoxylin_sigma_range,
            haematoxylin_bias_range,
            eosin_sigma_range,
            eosin_bias_range,
            dab_sigma_range,
            dab_bias_range,
            cutoff_range,
        )

    def get_transform_init_args_names(self):
        return "h_sigma", "e_sigma"

    def apply(self, img, **params):
        self.hed_augmentor.randomize()
        return np.array(self.hed_augmentor.transform(img)).astype(np.uint8)


class StainAugmentor(ImageOnlyTransform):
    def __init__(
        self,
        luminosity=True,
        method="macenko",
        sigma1=0.2,
        sigma2=0.2,
        augment_background=False,
        always_apply=False,
        p=0.5,
    ):
        """
        Augment the staining of H&E histology slides.
        This function augments the staining of H&E histology slides.
        Args:
            luminosity: Luminosity normalization.
            method: 'macenko', 'vahadane'
        Returns:
            Normalized (transformed) image.
        """
        super(StainAugmentor, self).__init__(always_apply, p)
        self.method = method
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.luminosity = luminosity
        self.augmentor = StainlibStainAugmentor(
            method=method,
            sigma1=sigma1,
            sigma2=sigma2,
            augment_background=augment_background,
        )

    def get_transform_init_args_names(self):
        return "luminosity", "method", "sigma1", "sigma2"

    def apply(self, img, **params):
        """
        Augment image
        Args:
            img: image to augment
        """
        to_augment = img.copy()
        if self.luminosity:
            # Standardize brightness (optional, can improve the tissue mask calculation)
            to_augment = LuminosityStandardizer.standardize(to_augment)
        self.fit(to_augment)

        return np.array(self.pop()).astype(np.uint8)
