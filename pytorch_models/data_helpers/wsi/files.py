from pathlib import Path
from typing import Union
from sourcelib.file import File
from wholeslidedata.data.extensions import WHOLE_SLIDE_IMAGE_EXTENSIONS
from wholeslidedata.data.mode import WholeSlideMode
from .wholeslideimage import MultiResWholeSlideImage


class MultiResWholeSlideImageFile(File):
    EXTENSIONS = WHOLE_SLIDE_IMAGE_EXTENSIONS
    IDENTIFIER = "mrwsi"

    def __init__(
        self,
        mode: Union[str, WholeSlideMode],
        path: Union[str, Path],
        image_backend: str = None,
    ):
        super().__init__(path=path, mode=mode)
        self._image_backend = image_backend

    def copy(self, destination_folder) -> None:
        destination_folder = Path(destination_folder) / "images"
        super().copy(destination_folder=destination_folder)

    def open(
        self,
        cell_graph_extractor: str = None,
        cell_graph_image_normalizer: str = "vahadane",
    ):
        return MultiResWholeSlideImage(
            self.path,
            self._image_backend,
            cell_graph_extractor=cell_graph_extractor,
            cell_graph_image_normalizer=cell_graph_image_normalizer,
        )
