import sys
from setuptools import setup

setup(
    name="pytorch_models",
    version="0.1.0",
    packages=[
        "pytorch_models",
        "pytorch_models.post",
        "pytorch_models.post.distances",
        "pytorch_models.optim",
        "pytorch_models.utils",
        "pytorch_models.utils.metrics",
        "pytorch_models.losses",
        "pytorch_models.models",
        "pytorch_models.models.mae",
        "pytorch_models.models.graph",
        "pytorch_models.models.segmentation",
        "pytorch_models.models.segmentation.unet",
        "pytorch_models.models.segmentation.deeplab",
        "pytorch_models.models.segmentation.my_models",
        "pytorch_models.models.ssl_features",
        "pytorch_models.models.classification",
        "pytorch_models.data_helpers",
    ],
    url="https://github.com/tsikup/pytorch_models",
    license="MIT",
    author="Nikos Tsiknakis",
    author_email="tsiknakisn@gmail.com",
    description=" A collection of pytorch models and utility methods that I use for any deep-learning project. ",
    install_requires=[
        "dotmap",
        "natsort",
        "numpy",
        "pandas",
        "torch",
        "tqdm",
        "opencv-python",
        "scikit-image",
        "Pillow",
        "shapely",
        "torch",
        "torchvision",
        "pytorch_lightning",
        "torchmetrics",
        "pydensecrf",
        "scipy",
        "Ranger21",
        "timm",
        "pytorch-lightning-bolts",
        "networkx",
        "dgl",
        "topk",
        "albumentations",
        "stainlib",
        "histocartography",
        "wholeslidedata",
        "shapely",
    ]
    + (["spams"] if sys.platform == "darwin" else ["spams-bin"]),
)
