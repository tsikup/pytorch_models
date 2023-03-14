import glob
import os
import dgl
import h5py
import torch
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import albumentations as A
from PIL import Image
from pathlib import Path
from dotmap import DotMap
from dgl import load_graphs
from typing import Dict, Union, Tuple, List
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .data_augmentors import get_augmentor
from ..utils.tensor import UnNormalize, tensor_to_image


class DataModuleHDF5(pl.LightningDataModule):
    def __init__(
        self,
        config: DotMap,
        data_dir: str,
        data_name: str,
        split: str = "train",
        val_data_name: str = None,
        imagenet_norm: bool = True,
        transform: Union[A.Compose, str] = None,
        multiresolution: bool = False,
        segmentation: bool = True,
        normalize: bool = True,
        mean: Union[List[float], Tuple[float]] = None,
        std: Union[List[float], Tuple[float]] = None,
        normalization_csv_dir: Union[str, Path] = None,
        normalization_csv_name: str = None,
    ):
        """
        Args:
            config: DotMap config file
            data_dir: directory containing the data
            data_name: name of the hdf5 file
            split: training or testing split
            val_data_name: name of the validation hdf5 file
            imagenet_norm: use normalization mean and std values from imagenet
            transform: transforms to use, else use default
        """
        super().__init__()
        self.split = split
        self.config = config
        self.data_dir = data_dir  # where you put the data
        self.data_name = data_name
        self.val_data_name = val_data_name
        self.targets = self.config.model.output_shape
        self.shape = tuple(self.config.model.input_shape)
        self.batch_size = self.config.trainer.batch_size
        self.shuffle = self.config.trainer.shuffle
        self.data_cols = self.config.dataset.data_cols.toDict()
        self.multiresolution = multiresolution
        self.segmentation = segmentation
        self.normalization_csv_dir = normalization_csv_dir
        self.normalization_csv_name = normalization_csv_name
        self.num_workers = self.config.trainer.num_workers
        self.prefetch_factor = self.config.trainer.prefetch_factor
        self.persistent_workers = self.config.trainer.persistent_workers
        self.normalize = normalize

        self.mean = mean
        self.std = std

        self.train = None
        self.val = None
        self.test = None

        if self.mean is None or self.std is None:
            if not imagenet_norm:
                try:
                    normalization_csv = pd.read_csv(
                        os.path.join(
                            self.normalization_csv_dir,
                            self.normalization_csv_name,
                        )
                    )
                    mean = normalization_csv[["mean_R", "mean_G", "mean_B"]]
                    std = normalization_csv[["std_R", "std_G", "std_B"]]
                    mean = np.squeeze(mean.to_numpy()).tolist()
                    std = np.squeeze(std.to_numpy()).tolist()
                    self.mean = mean
                    self.std = std
                except Exception as e:
                    print(
                        f"An exception has occured when loading csv file with mean and std values: {e}. Using default imagenet mean and std values."
                    )
                    self.mean = [0.485, 0.456, 0.406]
                    self.std = [0.229, 0.224, 0.225]
            else:
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]

        self.transform = (
            transform
            if (transform is not None and transform != "default")
            else self.get_default_transform()
        )
        self.pytorch_transform = self.get_pytorch_transform()

    def get_mean_std(self):
        return self.mean, self.std

    def get_default_transform(self):
        return {
            "train": get_augmentor(
                split="train",
                patch_size=self.config.model.input_shape[1],
                enable_augmentation=self.config.augment.enable,
                enable_stain_augmentation=self.config.augment.stain_augmentation.enable,
                replace_background=self.config.augment.replace_black_with_white_background_online,
                constant_pad_value=self.config.preprocess.filters2apply.constant_pad_value,
            ),
            "val": get_augmentor(
                split="val",
                patch_size=self.config.model.input_shape[1],
                enable_augmentation=False,
                enable_stain_augmentation=False,
                replace_background=self.config.augment.replace_black_with_white_background_online,
                constant_pad_value=self.config.preprocess.filters2apply.constant_pad_value,
            ),
            "test": get_augmentor(
                split="val",
                patch_size=self.config.model.input_shape[1],
                enable_augmentation=False,
                enable_stain_augmentation=False,
                replace_background=self.config.augment.replace_black_with_white_background_online,
                constant_pad_value=self.config.preprocess.filters2apply.constant_pad_value,
            ),
        }

    def get_pytorch_transform(self):
        pytorch_transform = [T.ToTensor()]
        if self.normalize:
            pytorch_transform.append(
                T.Normalize(
                    mean=self.mean,
                    std=self.std,
                )
            )
        return T.Compose(pytorch_transform)

    def get_transform(self, mode="train"):
        if self.transform is not None:
            if mode in list(self.transform.keys()):
                return self.transform[mode]
        return None

    def setup(self, stage=None):
        dataset_kargs = {}
        if self.multiresolution:
            dataset_cls = MultiResDatasetHDF5
            dataset_kargs = {
                "return_graph_path": False,
                "recalculate_graph": False,
                "segmentation": self.segmentation,
            }
        elif self.segmentation:
            dataset_cls = SegmentationDatasetHDF5
        else:
            dataset_cls = ClassificationDatasetHDF5

        if stage == "fit" or stage == "train" or stage is None:
            try:
                self.train = dataset_cls(
                    data_dir=self.data_dir,
                    data_name=self.data_name,
                    transform=self.get_transform("train"),
                    pytorch_transform=self.pytorch_transform,
                    data_cols=self.data_cols,
                    channels_last=False,
                    **dataset_kargs,
                )
                self.val = dataset_cls(
                    data_dir=self.data_dir,
                    data_name=self.val_data_name,
                    transform=self.get_transform("val"),
                    pytorch_transform=self.pytorch_transform,
                    data_cols=self.data_cols,
                    channels_last=False,
                    **dataset_kargs,
                )
            except AssertionError:
                print("Dataset not found. Using FakeDataset class.")
                self.train = FakeDataset(
                    input_shape=self.config.model.input_shape,
                    output_shape=self.config.model.output_shape,
                    classes=self.config.dataset.classes,
                )
                self.val = FakeDataset(
                    input_shape=self.config.model.input_shape,
                    output_shape=self.config.model.output_shape,
                    classes=self.config.dataset.classes,
                )
        elif stage == "test":
            self.test = dataset_cls(
                self.data_dir,
                self.data_name,
                self.get_transform("test"),
                pytorch_transform=self.pytorch_transform,
                data_cols=self.data_cols,
                channels_last=False,
                **dataset_kargs,
            )

    def get_label_distributions(self, mode="fit"):
        label_names = {
            key: value
            for key, value in zip(
                self.config.dataset.classes, self.config.dataset.target_names
            )
        }
        if mode in ["fit", "train"]:
            return {
                "train": self.train.get_label_distribution(
                    replace_names=label_names, as_figure=True
                ),
                "val": self.val.get_label_distribution(
                    replace_names=label_names, as_figure=True
                ),
            }
        elif mode in ["test", "eval"]:
            return {
                "test": self.test.get_label_distribution(
                    replace_names=label_names, as_figure=True
                )
            }

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = (
            dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        )
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def show_batch(self, win_size=(10, 10)):
        def _to_vis(data):
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        imgs, labels = next(iter(self.train_dataloader()))
        if self.normalize:
            unnormalized_imgs = torch.zeros_like(imgs)
            unorm = UnNormalize(mean=self.mean, std=self.std)
            for idx, img in enumerate(imgs):
                unnormalized_imgs[idx, ...] = unorm(img)
            imgs = unnormalized_imgs
        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))


class MultiResDatasetHDF5(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str] = None,
        transform=[ToTensorV2()],
        channels_last: bool = False,
        segmentation: bool = False,
        return_graph_path: bool = True,
        recalculate_graph: bool = False,
    ):
        """
        :param data_dir: hdf5 folder
        :param data_name: hdf5 filename
        :param transform: image transform pipeline
        """
        if data_cols is None and segmentation:
            data_cols = {
                "x_target": "x_target",
                "x_context": "x_context",
                "x_graph_path": "x_graph_path",
                "y_target": "y_target",
                "y_context": "y_context",
            }
        elif data_cols is None:
            data_cols = {
                "x_target": "x_target",
                "x_context": "x_context",
                "x_graph_path": "x_graph_path",
                "labels": "labels",
            }

        self.segmentation = segmentation
        self.return_graph_path = return_graph_path
        self.recalculate_graph = recalculate_graph

        self.h5_dataset = None
        self.x_target = None
        self.x_context = None
        self.x_graph_path = None
        if self.segmentation:
            self.y_target = None
            self.y_context = None
        else:
            self.labels = None
        self.graph_path = None
        self.cell_graphs = None

        # Return tensor with channels at last index
        self.channels_last = channels_last

        self.data_cols = data_cols
        self.h5_path = os.path.join(data_dir, data_name)
        # self.graph_path = os.path.join(data_dir, Path(data_name).stem + '_graphs' + '.bin')
        assert os.path.isdir(data_dir)
        assert os.path.exists(self.h5_path)
        # determine dataset length and shape
        with h5py.File(self.h5_path, "r") as f:
            # Total number of datapoints
            self.dataset_size = f[self.data_cols["x_target"]].shape[0]
            self.image_shape = f[self.data_cols["x_target"]].shape[1:]
            if self.segmentation:
                self.labels_shape = f[self.data_cols["y_target"]].shape[1:]
            else:
                self.labels_shape = [
                    1,
                ]

        # Albumentations transformation pipeline for the image (normalizing, etc.)
        self.transform = (
            A.Compose(
                [t for t in transform],
                additional_targets={"image1": "image", "mask1": "mask"},
            )
            if transform is not None
            else None
        )

    def get_graph_path(self, graph_path):
        return graph_path.decode("utf8").split(" idx=")[0]

    def get_graph_idx(self, graph_path):
        return int(graph_path.decode("utf8").split(" idx=")[1])

    def open_hdf5(self):
        # Open hdf5 file where images and labels are stored
        self.h5_dataset = h5py.File(self.h5_path, "r")
        self.x_target = self.h5_dataset[self.data_cols["x_target"]]
        self.x_context = self.h5_dataset[self.data_cols["x_context"]]
        self.x_graph_path = self.h5_dataset[self.data_cols["x_graph_path"]]
        if self.segmentation:
            self.y_target = self.h5_dataset[self.data_cols["y_target"]]
            self.y_context = self.h5_dataset[self.data_cols["y_context"]]
        else:
            if self.data_cols["labels"] is not None:
                self.labels = self.h5_dataset[self.data_cols["labels"]]
            else:
                self.labels = [None for _ in range(self.x_target.shape[0])]
        self.graph_path = self.get_graph_path(self.x_graph_path[0])
        if not self.return_graph_path:
            self.cell_graphs = dgl.load_graphs(self.graph_path)

    def get_label_distribution(self, replace_names: Dict = None, as_figure=False):
        if self.data_cols["labels"] is not None:
            label_dist = None
            with h5py.File(self.h5_path, "r") as f:
                labels = f[self.data_cols["labels"]][...]
                if as_figure:
                    labels = pd.DataFrame(labels, columns=["label"])
                    if replace_names is not None:
                        labels.replace(replace_names, inplace=True)
                    fig = sns.displot(labels, x="label", shrink=0.8)
                    label_dist = fig.figure
                else:
                    label_dist = np.unique(labels, return_counts=True)
            return label_dist
        return None

    def get_item(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        if not hasattr(self, "h5_dataset") or self.h5_dataset is None:
            self.open_hdf5()

        x_target = self.x_target[i]
        x_context = self.x_context[i]
        if self.segmentation:
            y_target = self.y_target[i]
            y_context = self.y_context[i]
        else:
            if self.data_cols["labels"] is not None:
                label = self.labels[i]
            else:
                label = None

        if not self.return_graph_path:
            x_graph = self.cell_graphs[self.get_graph_idx(self.x_graph_path[i])]
        else:
            if not self.recalculate_graph:
                x_graph = self.x_graph_path[i].decode("utf8")
            else:
                # TODO: IMPLEMENT GRAPH EXTRACTION
                pass

        if self.segmentation and len(y_target.shape) == 2:
            y_target = np.expand_dims(y_target, axis=-1)

        if self.segmentation and len(y_context.shape) == 2:
            y_context = np.expand_dims(y_context, axis=-1)

        if self.transform is not None:
            if self.segmentation:
                transformed = self.transform(
                    image=x_target, mask=y_target, image1=x_context, mask1=y_context
                )
                x_target = transformed["image"]
                y_target = transformed["mask"]

                x_context = transformed["image1"]
                y_context = transformed["mask1"]
            else:
                transformed = self.transform(image=x_target, image1=x_context)
                x_target = transformed["image"]
                x_context = transformed["image1"]

        # By default, ToTensorV2 returns C,H,W image and H,W,C mask
        if self.channels_last:
            x_target = x_target.permute(1, 2, 0)
            x_context = x_context.permute(1, 2, 0)
        else:
            if self.segmentation:
                y_target = y_target.permute(2, 0, 1)
                y_context = y_context.permute(2, 0, 1)

        if self.segmentation:
            return {
                "x_target": x_target,
                "x_context": x_context,
                "x_graph": x_graph,
                "y_target": y_target,
                "y_context": y_context,
            }
        else:
            return {
                "x_target": x_target,
                "x_context": x_context,
                "x_graph": x_graph,
                "label": label,
            }

    def __len__(self):
        return self.dataset_size


class DatasetHDF5(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last: bool = False,
        segmentation: bool = True,
        image_only: bool = False,
    ):
        """
        :param data_dir: hdf5 folder
        :param data_name: hdf5 filename
        :param transform: image transform pipeline
        """
        if data_cols is None:
            data_cols = {"images": "images", "labels": "masks"}

        self.segmentation = segmentation
        self.image_only = image_only

        self.h5_dataset = None
        self.images = None
        self.labels = None
        self.cell_graphs = None

        # Return tensor with channels at last index
        self.channels_last = channels_last

        self.data_cols = data_cols
        self.h5_path = os.path.join(data_dir, data_name)
        # self.graph_path = os.path.join(data_dir, Path(data_name).stem + '_graphs' + '.bin')
        assert os.path.isdir(data_dir)
        assert os.path.exists(self.h5_path)
        # determine dataset length and shape
        with h5py.File(self.h5_path, "r") as f:
            # Total number of datapoints
            self.dataset_size = f[self.data_cols["images"]].shape[0]
            self.image_shape = f[self.data_cols["images"]].shape[1:]
            if not self.image_only:
                self.labels_shape = (
                    f[self.data_cols["labels"]].shape[1:] if self.segmentation else 1
                )

        # Albumentations transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        # Pytorch transformation pipeline for the image
        self.pytorch_transform = pytorch_transform

    def open_hdf5(self):
        # Open hdf5 file where images and labels are stored
        self.h5_dataset = h5py.File(self.h5_path, "r")
        self.images = self.h5_dataset[self.data_cols["images"]]
        if not self.image_only:
            if self.data_cols["labels"] is not None:
                self.labels = self.h5_dataset[self.data_cols["labels"]]
        # if self.data_cols["cell_graphs"]:
        #     self.cell_graphs = dgl.load_graphs(self.graph_path)

    def get_label_distribution(self, replace_names: Dict = None, as_figure=False):
        if self.data_cols["labels"] is not None:
            label_dist = None
            with h5py.File(self.h5_path, "r") as f:
                labels = f[self.data_cols["labels"]][...]
                if as_figure:
                    labels = pd.DataFrame(labels, columns=["label"])
                    if replace_names is not None:
                        labels.replace(replace_names, inplace=True)
                    fig = sns.displot(labels, x="label", shrink=0.8)
                    label_dist = fig.figure
                else:
                    label_dist = np.unique(labels, return_counts=True)
            return label_dist
        return None

    def get_item(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        if not hasattr(self, "h5_dataset") or self.h5_dataset is None:
            self.open_hdf5()

        image = self.images[i]
        if not self.image_only:
            try:
                label = np.array([self.labels[i]])
            except IndexError:
                label = np.array([self.labels[0]])

        if not self.image_only and self.segmentation and len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)

        if self.segmentation and not self.image_only:
            if self.transform is not None:
                transformed = self.transform(image=image, mask=label)
                image = transformed["image"]
                label = transformed["mask"]
            image, label = self.pytorch_transform(img=image, mask=label)
        else:
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            if not self.image_only:
                label = torch.from_numpy(label)
            if self.pytorch_transform is not None:
                image = self.pytorch_transform(image)

        # By default, ToTensorV2 returns C,H,W image and H,W,C mask
        if self.channels_last:
            image = image.permute(1, 2, 0)
        else:
            if self.segmentation and not self.image_only:
                label = label.permute(2, 0, 1)

        if self.image_only:
            return image

        return image, label.to(torch.uint8)

    def __len__(self):
        return self.dataset_size


class ImageOnlyDatasetHDF5(DatasetHDF5):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last=False,
    ):
        super(ImageOnlyDatasetHDF5, self).__init__(
            data_dir=data_dir,
            data_name=data_name,
            data_cols=data_cols,
            transform=transform,
            pytorch_transform=pytorch_transform,
            channels_last=channels_last,
            image_only=True,
        )


class SegmentationDatasetHDF5(DatasetHDF5):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last=False,
    ):
        super(SegmentationDatasetHDF5, self).__init__(
            data_dir=data_dir,
            data_name=data_name,
            data_cols=data_cols,
            transform=transform,
            pytorch_transform=pytorch_transform,
            channels_last=channels_last,
            segmentation=True,
        )


class ClassificationDatasetHDF5(DatasetHDF5):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last=False,
    ):
        super(ClassificationDatasetHDF5, self).__init__(
            data_dir=data_dir,
            data_name=data_name,
            data_cols=data_cols,
            transform=transform,
            pytorch_transform=pytorch_transform,
            channels_last=channels_last,
            segmentation=False,
        )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        df,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: T.Compose,
        channels_last,
        return_slide_name,
    ):
        # transform should contain ToTensorV2 last
        if data_cols is None:
            data_cols = {
                "image_path": "image_path",
                "mask_path": "mask_path",
                "cell_graph_path": None,
            }
        assert isinstance(transform.transforms[-1], ToTensorV2) or isinstance(
            transform.transforms[-1], ToTensor
        )
        self.df = df
        self.data_cols = data_cols.cop
        self.transform = transform
        self.channels_last = channels_last
        self.return_slide_name = return_slide_name
        self.pytorch_transform = pytorch_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.df.iloc[idx][self.data_cols["image_path"]]))
        mask = np.array(Image.open(self.df.iloc[idx][self.data_cols["mask_path"]]))

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image, mask = self.pytorch_transform(image, mask)

        # By default, ToTensorV2 returns C,H,W image and H,W,C mask
        if self.channels_last:
            image = image.permute(1, 2, 0)
        else:
            mask = mask.permute(2, 0, 1)

        out = [image.float(), mask.to(torch.int64)]

        if self.data_cols["cell_graph_path"] is not None:
            cell_graph = load_graphs(
                self.df.iloc[idx][self.data_cols["cell_graph_path"]]
            )
            out.append(cell_graph)

        if self.return_slide_name:
            slide_name = self.df.iloc[idx]["slide"]
            return *out, slide_name

        else:
            return tuple(out)


class FakeDataset(Dataset):
    def __init__(
        self, input_shape=[3, 512, 512], output_shape=[1, 512, 512], classes=[0, 1]
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.classes = classes

    def __len__(self):
        return int(1e6)

    def __getitem__(self, item):
        img = torch.rand(*self.input_shape)
        label = np.random.randint(
            low=min(self.classes), high=max(self.classes) + 1, size=self.output_shape
        )
        return img, label


class FeatureDatasetOneHDF5(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        base_label: int = 0,
    ):
        """
        :param data_dir: hdf5 folder
        :param data_cols: hdf5 dataset names
        """
        self.base_label = base_label

        assert data_cols is not None
        self.data_cols = data_cols
        self.multiresolution = (
            "features_context" in self.data_cols
            and self.data_cols["features_context"] is not None
        )

        assert os.path.isdir(data_dir)
        self.h5_path = os.path.join(data_dir, data_name)
        assert os.path.isfile(self.h5_path)

        # determine dataset length and shape
        with h5py.File(self.h5_path, "r") as f:
            # Total number of datapoints
            self.dataset_size = f[self.data_cols["features_target"]].shape[0]
            self.features_shape = f[self.data_cols["features_target"]].shape[1]
            self.labels_shape = 1

        self.h5_dataset = None
        self.labels = None
        self.features = None
        self.features_context = None

    def open_hdf5(self):
        # Open hdf5 file where images and labels are stored
        self.h5_dataset = h5py.File(self.h5_path, "r")
        self.features = self.h5_dataset[self.data_cols["features_target"]]

        self.labels = self.h5_dataset[self.data_cols["labels"]][...] - self.base_label
        self.labels = np.array([self.labels]).flatten()

        features_context = None
        if self.multiresolution:
            self.features_context = self.h5_dataset[self.data_cols["features_context"]]

    def get_label_distribution(self, replace_names: Dict = None, as_figure=False):
        if as_figure:
            labels = pd.DataFrame(self.labels, columns=["label"])
            if replace_names is not None:
                labels.replace(replace_names, inplace=True)
            fig = sns.displot(labels, x="label", shrink=0.8)
            label_dist = fig.figure
        else:
            label_dist = np.unique(self.labels, return_counts=True)
        return label_dist

    def get_item(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        if not hasattr(self, "h5_dataset") or self.h5_dataset is None:
            self.open_hdf5()

        _features = torch.from_numpy(self.features[i])
        try:
            _label = torch.from_numpy(
                np.array([self.labels[i]], dtype=self.labels.dtype)
            )
        except IndexError:
            _label = torch.from_numpy(
                np.array([self.labels[0]], dtype=self.labels.dtype)
            )

        if self.multiresolution:
            _features_context = torch.from_numpy(self.features_context[i])
            return dict(features=_features, features_context=_features_context), _label
        return dict(features=_features), _label

    def __len__(self):
        return self.dataset_size

    @property
    def shape(self):
        return [self.dataset_size, self.features_shape]


class MILImageDatasetHDF5(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        data_dir: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        base_label: int = 0,
        num_tiles: int = -1,
        load_ram: bool = False,
    ):
        """
        :param data_dir: hdf5 folder
        :param data_cols: hdf5 dataset name
        """
        assert data_cols is not None

        self.data_cols = data_cols
        self.base_label = base_label
        self.num_tiles = num_tiles
        self.load_ram = load_ram

        self.transform = transform
        self.pytorch_transform = pytorch_transform

        self.multiresolution = (
            "x_context" in self.data_cols and self.data_cols["x_context"] is not None
        )

        self.h5_dataset = None
        self.labels = None

        assert os.path.isdir(data_dir)

        self.slides = glob.glob(os.path.join(data_dir, "*.h5"))

        # determine dataset length and shape
        self.dataset_size = len(self.slides)
        self.labels_shape = 1
        with h5py.File(self.slides[0], "r") as f:
            # Total number of datapoints
            self.image_shape = f[self.data_cols["x_target"]].shape[1:]

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.vstack(target)
        return [data, target]

    def open_hdf5(self, h5_path, load_ram=False):
        # Open hdf5 file where images and labels are stored
        _to_return = None
        import time

        start = time.time()
        with h5py.File(h5_path, "r") as h5_dataset:
            label = h5_dataset[self.data_cols["labels"]][0] - self.base_label
            label = torch.from_numpy(np.array([label], dtype=label.dtype))

            if load_ram:
                _images = h5_dataset[self.data_cols["x_target"]][...]
                if self.multiresolution:
                    _images_context = h5_dataset[self.data_cols["x_context"]][...]
            else:
                _images = h5_dataset[self.data_cols["x_target"]]
                if self.multiresolution:
                    _images_context = h5_dataset[self.data_cols["x_context"]]

            if self.num_tiles > 0:
                _indices = np.array(
                    np.sort(np.random.choice(list(range(len(_images))), self.num_tiles))
                )
            else:
                _indices = list(range(len(_images)))

            images = []
            images_context = [] if self.multiresolution else None

            if self.pytorch_transform is not None or self.transform is not None:
                for idx in _indices:
                    _img = _images[idx]

                    if self.multiresolution:
                        _img_context = _images_context[idx]

                    if self.transform is not None:
                        if not self.multiresolution:
                            _img = self.transform(image=_img)["image"]
                        else:
                            _transformed = self.transform(
                                image=_img, context=_img_context
                            )
                            _img = _transformed["image"]
                            _img_context = _transformed["context"]

                    if self.pytorch_transform is not None:
                        _img = self.pytorch_transform(_img)
                        if self.multiresolution:
                            _img_context = self.pytorch_transform(_img_context)

                    images.append(_img)
                    if self.multiresolution:
                        images_context.append(_img_context)

                images = torch.stack(images, dim=0)
                if self.multiresolution:
                    images_context = torch.stack(images_context, dim=0)
            else:
                images = _images

            if self.multiresolution:
                _to_return = dict(images=images, images_context=images_context), label

            _to_return = dict(images=images), label

        return _to_return

    def get_label_distribution(self, replace_names: Dict = None, as_figure=False):
        labels = []
        for slide in self.slides:
            with h5py.File(slide, "r") as f:
                label = f[self.data_cols["labels"]][0]
                labels.append(label)
        if as_figure:
            labels = pd.DataFrame(labels, columns=["label"])
            if replace_names is not None:
                labels.replace(replace_names, inplace=True)
            fig = sns.displot(labels, x="label", shrink=0.8)
            label_dist = fig.figure
        else:
            label_dist = np.unique(labels, return_counts=True)
        return label_dist

    def get_item(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        h5_path = self.slides[i]
        return self.open_hdf5(h5_path, load_ram=self.load_ram)

    def __len__(self):
        return self.dataset_size

    @property
    def shape(self):
        return self.image_shape
