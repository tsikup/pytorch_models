import numpy as np
from pathlib import Path
from typing import Union, Dict
from numpy import ndarray
from wholeslidedata import WholeSlideAnnotation
from wholeslidedata.image.backend import WholeSlideImageBackend
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation import utils as annotation_utils

from .annotation_parser import QuPathAnnotationParser
from ..graphs import CellGraphExtractor
import he_preprocessing.utils.image as ut_image


class MultiResWholeSlideImage(WholeSlideImage):
    def __init__(
        self,
        path: Union[Path, str],
        backend: Union[WholeSlideImageBackend, str] = "openslide",
        annotation_path=None,
        labels=None,
        cell_graph_extractor=None,
        cell_graph_image_normalizer="vahadane",
    ):
        super(MultiResWholeSlideImage, self).__init__(path=path, backend=backend)
        self.extract_graph = cell_graph_extractor is not None

        self.annotation = None
        if annotation_path:
            assert annotation_path.endswith(
                ".geojson"
            ), "Annotation file must be a QuPath geojson file."
            self._annotation_parser = QuPathAnnotationParser()
            self.annotation = WholeSlideAnnotation(
                annotation_path=annotation_path,
                labels=labels,
                parser=self._annotation_parser,
            )

        if cell_graph_extractor is not None:
            self.cg_extractor = CellGraphExtractor(
                feature_extractor=cell_graph_extractor,
                stain_norm=cell_graph_image_normalizer,
            )

    @property
    def labels(self):
        if self.annotation is not None:
            return self.annotation.labels
        return []

    @property
    def annotation_counts(self):
        return annotation_utils.get_counts_in_annotations(self.annotation.annotations)

    @property
    def annotations_per_label(self) -> Dict[str, int]:
        return annotation_utils.get_counts_in_annotations(
            self.annotation.annotations, labels=self.labels
        )

    @property
    def pixels_count(self):
        return annotation_utils.get_pixels_in_annotations(self.annotation.annotations)

    @property
    def pixels_per_label(self) -> Dict[str, int]:
        return annotation_utils.get_pixels_in_annotations(
            self.annotation.annotations, labels=self.labels
        )

    def get_tissue_mask(self, spacing=32, return_contours=True):
        # TODO: if annotation exists, use it to get the tissue mask
        downsample = self.get_downsampling_from_spacing(spacing)
        return ut_image.get_slide_tissue_mask(
            slide_path=self.path,
            downsample=downsample,
            return_contours=return_contours,
        )

    def get_num_details(self, width: int, height: int, spacings: Dict[str, float]):
        if "details" in spacings:
            downsampling_target = self.get_downsampling_from_spacing(
                spacing=spacings["target"]
            )
            downsampling_details = self.get_downsampling_from_spacing(
                spacing=spacings["details"]
            )
            # This is the relative image size of detail patches with respect to the target resolution.
            details_width = int(
                width / (int(downsampling_target) / int(downsampling_details))
            )
            details_height = int(
                height / (int(downsampling_target) / int(downsampling_details))
            )

            assert width % details_width == 0 and height % details_height == 0, (
                "The relative size of detail patches must divide tile size "
                "perfectly. E.g. target size -> 512, then detail_size -> ["
                "256, 128, 64, ...] "
            )

            # Number of detail patches that will be extracted from the target patch at a higher resolution.
            num_details_patches = int(width // details_width) * int(
                height // details_height
            )
            # Total iterations over the x-axis to get all detail patches
            total_i = int(width / details_width)
            # Total iterations over the y-axis to get all detail patches
            total_j = int(height / details_height)

            return num_details_patches, total_i, total_j
        return None, None, None

    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacings: Dict[str, float],
        center: bool = True,
        relative: bool = False,
        tissue_percentage: float = None,
        blurriness_threshold: Dict[str, int] = None,
    ) -> Union[
        dict[str, Union[ndarray, ndarray, list[Union[ndarray, ndarray]]]],
        dict[str, Union[ndarray, ndarray]],
    ]:
        """Extracts multi-resolution patches/regions from the wholeslideimage
        Args:
            x (int): x value
            y (int): y value
            width (int): width of region
            height (int): height of region
            spacings (list of float): spacing/resolution of the patch at target, context and details level
            center (bool, optional): if x,y values are centres or top left coordinated. Defaults to True.
            relative (bool, optional): if x,y values are a reference to the dimensions of the specified spacing.
                                       Defaults to False.
            tissue_percentage (int): if target image has tissue percentage lower than that, discard
            blurriness_threshold (dict): thresholds to detect if target or context image are blurry
        Returns:
            np.ndarray: numpy patch
        """

        assert "target" in spacings and (
            "context" in spacings or "graph" in spacings or "details" in spacings
        ), (
            "Spacings should be a dict with the following keys: 'target', 'context' (optional), 'graph' (optional), "
            "'details' (optional), "
        )

        _spacings = spacings.copy()

        for key, value in _spacings.items():
            _spacings[key] = self.get_real_spacing(value)

        # Target
        x_target = self.get_patch(
            x,
            y,
            width,
            height,
            spacing=_spacings["target"],
            center=center,
            relative=relative,
        )

        # Context
        x_context = self.get_patch(
            x,
            y,
            width,
            height,
            spacing=_spacings["context"],
            center=center,
            relative=relative,
        )

        # Graph
        x_graph = None
        cell_graph = None
        if self.extract_graph:
            # Possibly discard if blurry or not enough tissue to save time from graph computation
            if (
                (
                    tissue_percentage is None
                    or ut_image.keep_tile(
                        x_target,
                        width,
                        tissue_threshold=tissue_percentage,
                        roi_mask=None,
                        pad=True,
                    )
                )
                and (
                    blurriness_threshold["target"] is None
                    or (
                        not ut_image.is_blurry(
                            x_target,
                            threshold=blurriness_threshold["target"],
                            normalize=True,
                            verbose=0,
                        )
                    )
                )
                and (
                    blurriness_threshold["context"] is None
                    or (
                        not ut_image.is_blurry(
                            x_context,
                            threshold=blurriness_threshold["context"],
                            normalize=True,
                            verbose=0,
                        )
                    )
                )
            ):
                # Image to build graph on
                x_graph = self.get_patch(
                    x,
                    y,
                    width=int(width * _spacings["target"] / _spacings["graph"]),
                    height=int(height * _spacings["target"] / _spacings["graph"]),
                    spacing=_spacings["graph"],
                    center=center,
                    relative=relative,
                )
                try:
                    cell_graph = self.cg_extractor.process(x_graph)
                except ValueError:
                    cell_graph = None
            else:
                print(
                    "Target tile doesn't contain enough tissue or it's blurry. Don't extract cell graph."
                )

        # Details
        x_details = None
        if "details" in _spacings:
            x_details = []

            num_details_patches, total_i, total_j = self.get_num_details(
                width, height, _spacings
            )

            if center:
                # Get top left coords of target resolution
                downsampling = int(
                    self.get_downsampling_from_spacing(_spacings["target"])
                )
                x, y = x - downsampling * (width // 2), y - downsampling * (height // 2)

            for idx in range(self.num_details_patches):
                i = int(idx // self.total_i)
                j = int(idx % self.total_j)

                rel_coord_x = j * self.details_size * self.downsampling_target
                rel_coord_y = i * self.details_size * self.downsampling_target

                coord_x = x + rel_coord_x
                coord_y = y + rel_coord_y
                x_details.append(
                    self.get_patch(
                        coord_x,
                        coord_y,
                        width,
                        height,
                        spacing=_spacings["details"],
                        center=False,
                        relative=relative,
                    )
                )

        out = {
            "target": x_target,
            "details": x_details,
            "context": x_context,
            "graph_image": x_graph,
            "graph": cell_graph,
        }

        return {k: v for k, v in out.items() if v is not None}
