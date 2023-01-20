import numpy as np
from pathlib import Path
from natsort import os_sorted
from PIL import Image, ImageDraw
from typing import List, Union, Dict

from wholeslidedata.annotation.parser import AnnotationParser
from wholeslidedata.annotation.structures import Annotation
from wholeslidedata.samplers.annotationsampler import OrderedAnnotationSampler
from wholeslidedata.samplers.batchsampler import BatchSampler
from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler
from wholeslidedata.samplers.patchsampler import PatchSampler
from wholeslidedata.samplers.pointsampler import CenterPointSampler
from wholeslidedata.samplers.samplesampler import SampleSampler
from wholeslidedata.samplers.structures import BatchShape
from wholeslidedata.source.associations import associate_files
from wholeslidedata.source.files import WholeSlideFile
from wholeslidedata.source.utils import (
    NoSourceFilesInFolderError,
    factory_sources_from_paths,
)

from . import (
    QuPathAnnotationParser,
    MultiResWholeSlideDataSet,
    MultiResWholeSlideImageFile,
    MaskedTiledAnnotationHook,
    BatchOneTimeReferenceSampler,
    OrderedLabelOneTimeSampler,
    MultiResSampleSampler,
    MultiResPatchSampler,
)


def create_batch_sampler(
    slides_dir: Union[str, Path] = None,
    annotations_dir: Union[str, Path] = None,
    image_files: List[Union[str, Path]] = None,
    annotation_files: List[Union[str, Path]] = None,
    slide_extension=".ndpi",
    ann_extension=".geojson",
    file_type: Union[str, type] = "mrwsi",
    tile_size: int = 512,
    tissue_percentage: float = 0.5,
    stride_overlap_percentage: float = 0.0,
    intersection_percentage: float = 1.0,
    blurriness_threshold: Dict[str, int] = None,
    batch_size: int = 1,
    labels: Union[Dict[str, int], None] = None,
    spacing: Union[Dict[str, float], None] = None,
    seed=123,
):
    if labels is None:
        labels = dict(tissue=0, tumor=1)

    if spacing is None:
        spacing = dict(target=0.5, context=2.0, graph=0.25)

    if blurriness_threshold is None:
        blurriness_threshold = dict(target=500, context=5000)

    if image_files is None:
        assert (
            slides_dir is not None and annotations_dir is not None
        ), "If 'image_files' is None then 'slides_dir' and 'annotations_dir' should be provided."
        image_files = whole_slide_files_from_folder_factory(
            slides_dir,
            file_type,
            excludes=[
                "mask",
            ],
            filters=[
                slide_extension,
            ],
            image_backend="openslide",
        )

        if ann_extension == ".geojson":
            parser = QuPathAnnotationParser
        else:
            parser = AnnotationParser
        parser = parser(
            labels={"tumor": 1, "tissue": 0},
            hooks=(
                MaskedTiledAnnotationHook(
                    tile_size=tile_size,
                    ratio=1,
                    overlap=int(tile_size * stride_overlap_percentage),
                    intersection_percentage=intersection_percentage,
                    label_names=list(labels.keys()),
                    full_coverage=True,
                ),
            ),
        )
        annotation_files = whole_slide_files_from_folder_factory(
            annotations_dir,
            "wsa",
            excludes=["tif"],
            filters=[ann_extension],
            annotation_parser=parser,
        )
    else:
        assert annotation_files is not None

    associations = associate_files(image_files, annotation_files)

    dataset = MultiResWholeSlideDataSet(
        mode="default",
        associations=associations,
        labels=list(labels.keys()),
        cell_graph_extractor="resnet34" if "graph" in spacing else None,
        cell_graph_image_normalizer="vahadane",
    )

    batch_ref_sampler = BatchOneTimeReferenceSampler(
        dataset=dataset,
        batch_size=batch_size,
        label_sampler=OrderedLabelOneTimeSampler(
            annotations_per_label=dataset.annotations_per_label, seed=seed
        ),
        annotation_sampler=OrderedAnnotationSampler(
            dataset.annotations_per_label, seed=seed
        ),
        point_sampler=CenterPointSampler(dataset=dataset, seed=seed),
    )

    batch_shape = BatchShape(
        batch_size,
        spacing=[(key, value) for key, value in spacing.items()],
        shape=[[tile_size, tile_size, 3] for _ in spacing],
        labels=dataset.sample_labels,
    )

    if file_type == "mrwsi":
        sample_sampler = MultiResSampleSampler(
            patch_sampler=MultiResPatchSampler(
                center=True,
                relative=False,
                tissue_percentage=tissue_percentage,
                blurriness_threshold=blurriness_threshold,
            ),
            patch_label_sampler=SegmentationPatchLabelSampler(),
            batch_shape=batch_shape,
        )
    else:
        sample_sampler = SampleSampler(
            patch_sampler=PatchSampler(center=True, relative=False),
            patch_label_sampler=SegmentationPatchLabelSampler(),
            batch_shape=batch_shape,
        )

    batch_sampler = BatchSampler(dataset=dataset, sampler=sample_sampler)

    return batch_sampler, batch_ref_sampler, batch_shape


def whole_slide_files_from_folder_factory(
    folder: Union[str, Path],
    file_type: Union[str, type],
    mode: str = "default",
    filters: List[str] = (),
    excludes: List[str] = (),
    recursive=False,
    **kwargs,
):
    if file_type == "mrwsi":
        class_type = MultiResWholeSlideImageFile
    else:
        class_type = WholeSlideFile.get_registrant(file_type)
    all_sources = []
    folder = Path(folder)
    for extension in class_type.EXTENSIONS.names():
        paths = os_sorted(
            folder.rglob("*" + extension) if recursive else folder.glob("*" + extension)
        )
        sources = factory_sources_from_paths(
            class_type, mode, paths, filters, excludes, **kwargs
        )
        all_sources.extend(sources)

    if all_sources == []:
        raise NoSourceFilesInFolderError(class_type, filters, excludes, folder)
    return all_sources


def get_annotation_mask(
    image: Union[Image.Image, np.ndarray],
    annotations: List[Annotation],
    scale=1.0,
    relative_bounds: List[int] = [0, 0],
):
    return draw_relative_annotations(
        image,
        annotations,
        scale=scale,
        relative_bounds=relative_bounds,
        outline_color=(255, 255, 255),
        fill_color=(255, 255, 255),
        width=1,
        mask_only=True,
    )


def draw_relative_annotations(
    image: Image.Image,
    annotations: List[Annotation],
    use_base_coordinates=False,
    scale=1.0,
    relative_bounds: List[int] = [0, 0],
    plot_mask=False,
    outline_color=(255, 0, 0),
    fill_color=None,
    width=3,
    mask_only=False,
):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if mask_only:
        _image = Image.new("RGB", image.size)
    else:
        _image = image.copy()

    for annotation in annotations:
        if plot_mask:
            coordinates = annotation.mask_coordinates
        elif use_base_coordinates:
            coordinates = [annotation.base_coordinates]
        else:
            coordinates = [annotation.coordinates]

        if plot_mask and use_base_coordinates:
            coordinates = [coordinates - annotation.bounds[:2]]
        else:
            coordinates = [
                (coord - np.array(relative_bounds)) * scale for coord in coordinates
            ]

        if annotation.type == "point":
            draw = ImageDraw.Draw(_image)
            for coord in coordinates:
                #     ax.scatter(*coord, color=annotation.label.color)
                draw.point(coord, fill=fill_color)
        elif annotation.type == "polygon":
            draw = ImageDraw.Draw(_image)
            for coord in coordinates:
                # ax.plot(*list(zip(*coord)), color=color, linewidth=2)
                xy = [tuple(_xy) for _xy in coord]
                draw.polygon(
                    xy=xy,
                    outline=outline_color if outline_color else None,
                    fill=fill_color if fill_color else None,
                    width=width,
                )
        else:
            raise ValueError(f"invalid annotation {type(annotation)}")
    # if ax == plt:
    #     plt.axis("equal")
    #     plt.show()
    # else:
    #     ax.axis("equal")
    #     ax.set_title(title)
    return _image
