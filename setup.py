import sys
from setuptools import setup, find_packages

EXTRA_REQUIREMENTS = {
    "graphs": ["networkx", "dgl", "histocartography"],
}

setup(
    name="pytorch_models",
    version="0.1.0",
    packages=find_packages(),
    url="https://github.com/tsikup/pytorch_models",
    license="MIT",
    author="Nikos Tsiknakis",
    author_email="tsiknakisn@gmail.com",
    description=" A collection of pytorch models and utility methods that I use for any deep-learning project. ",
    extras_require=EXTRA_REQUIREMENTS,
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
        "lightning",
        "torchmetrics",
        "pydensecrf",
        "scipy",
        "Ranger21",
        "timm",
        "pytorch-lightning-bolts",
        "topk",
        "albumentations",
        "stainlib",
        "wholeslidedata",
        "shapely",
    ]
    + (["spams"] if sys.platform == "darwin" else ["spams-bin"]),
)
