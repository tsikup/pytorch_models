from pathlib import Path
from typing import Union
from wholeslidedata.extensions import (
    FolderCoupledExtension,
    WholeSlideImageExtension,
)
from wholeslidedata.mode import Mode
from wholeslidedata.source.copy import copy as copy_source
from wholeslidedata.source.files import WholeSlideFile, ImageFile

from .wholeslideimage import MultiResWholeSlideImage


@WholeSlideFile.register(
    ("mrwsi", "multires_wsi", "multiresolutionwsi", "multiresolutionwholeslideimage")
)
class MultiResWholeSlideImageFile(WholeSlideFile, ImageFile):
    EXTENSIONS = WholeSlideImageExtension

    def __init__(
        self, mode: Union[str, Mode], path: Union[str, Path], image_backend: str = None
    ):
        super().__init__(mode, path, image_backend)

    def copy(self, destination_folder) -> None:
        destination_folder = Path(destination_folder) / "images"
        extension_name = self.path.suffix
        if WholeSlideImageExtension.is_extension(
            extension_name, FolderCoupledExtension
        ):
            folder = self.path.with_suffix("")
            copy_source(folder, destination_folder)
        super().copy(destination_folder=destination_folder)

    def open(
        self,
        cell_graph_extractor: str = "resnet34",
        cell_graph_image_normalizer: str = "vahadane",
    ):
        return MultiResWholeSlideImage(
            self.path,
            self._image_backend,
            cell_graph_extractor=cell_graph_extractor,
            cell_graph_image_normalizer=cell_graph_image_normalizer,
        )
