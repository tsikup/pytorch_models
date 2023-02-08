from typing import Dict, Tuple
from wholeslidedata.annotation import utils as annotation_utils
from wholeslidedata.dataset import DataSet, WholeSlideSampleReference
from wholeslidedata.labels import Labels
from wholeslidedata.source.associations import Associations
from wholeslidedata.source.files import WholeSlideAnnotationFile

from .files import MultiResWholeSlideImageFile


class MultiResWholeSlideDataSet(DataSet):
    def __init__(
        self,
        mode,
        associations: Associations,
        labels: Labels = None,
        cell_graph_extractor: str = None,
        cell_graph_image_normalizer="vahadane",
        load_images=True,
        copy_path=None,
    ):
        self._load_images = load_images
        self._copy_path = copy_path
        self._cell_graph_extractor = cell_graph_extractor
        self._cell_graph_image_normalizer = cell_graph_image_normalizer
        super().__init__(mode, associations, labels)

    def _open(self, associations, labels):
        data = dict()
        for file_key, associated_files in associations.items():
            data[file_key] = {
                self.__class__.IMAGES_KEY: dict(),
                self.__class__.ANNOTATIONS_KEY: dict(),
            }
            for wsi_index, wsi_file in enumerate(
                associated_files[MultiResWholeSlideImageFile]
            ):
                data[file_key][self.__class__.IMAGES_KEY][wsi_index] = self._open_image(
                    wsi_file
                )

            for wsa_index, wsa_file in enumerate(
                associated_files[WholeSlideAnnotationFile]
            ):
                data[file_key][self.__class__.ANNOTATIONS_KEY][
                    wsa_index
                ] = self._open_annotation(wsa_file, labels=labels)
        return data

    def _open_image(self, wsi_file: MultiResWholeSlideImageFile):
        if self._copy_path:
            wsi_file.copy(self._copy_path)
        if self._load_images:
            return wsi_file.open(
                cell_graph_extractor=self._cell_graph_extractor,
                cell_graph_image_normalizer=self._cell_graph_image_normalizer,
            )
        return wsi_file

    def _open_annotation(self, wsa_file: WholeSlideAnnotationFile, labels):
        if self._copy_path:
            wsa_file.copy(self._copy_path)
        return wsa_file.open(labels=labels)

    def _init_labels(self):
        labels = []
        for values in self._data.values():
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                for annotation in wsa._annotations:
                    labels.append(annotation.label)
        return Labels.create(list(set(labels)))

    def _init_samples(self) -> Tuple:
        sample_references = {}
        for file_index, (file_key, values) in enumerate(self._data.items()):
            for wsa_index, wsa in values[self.__class__.ANNOTATIONS_KEY].items():
                for annotation in wsa.sampling_annotations:
                    sample_references.setdefault(annotation.label.name, []).append(
                        WholeSlideSampleReference(
                            file_index=file_index,
                            file_key=file_key,
                            wsa_index=wsa_index,
                            annotation_index=annotation.index,
                        )
                    )

        return sample_references

    def close_images(self):
        for image in self._images.values():
            image.close()
            del image
        self._images = {}

    @property
    def annotation_counts(self):
        _counts = []
        for values in self._data.values():
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                _counts.append(
                    annotation_utils.get_counts_in_annotations(wsa.annotations)
                )
        return sum(_counts)

    @property
    def annotations_per_label(self) -> Dict[str, int]:
        counts_per_label_ = {label.name: 0 for label in self._labels}
        for values in self._data.values():
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                for label, count in annotation_utils.get_counts_in_annotations(
                    wsa.annotations, labels=self._labels
                ).items():
                    if label in counts_per_label_:
                        counts_per_label_[label] += count
        return counts_per_label_

    @property
    def annotations_per_key(self):
        _counts_per_key = {}
        for file_key, values in self._data.items():
            _counts_per_key[file_key] = 0
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                _counts_per_key[file_key] += annotation_utils.get_counts_in_annotations(
                    wsa.annotations
                )
        return _counts_per_key

    @property
    def annotations_per_label_per_key(self):
        counts_per_label_per_key_ = {}
        for file_key, values in self._data.items():
            counts_per_label_per_key_[file_key] = {}
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                for label, count in annotation_utils.get_counts_in_annotations(
                    wsa.annotations, labels=self._labels
                ).items():
                    if label not in counts_per_label_per_key_[file_key]:
                        counts_per_label_per_key_[file_key][label] = 0
                    counts_per_label_per_key_[file_key][label] += count
        return counts_per_label_per_key_

    @property
    def pixels_count(self):
        _counts = []
        for values in self._data.values():
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                _counts.append(
                    annotation_utils.get_pixels_in_annotations(wsa.annotations)
                )
        return sum(_counts)

    @property
    def pixels_per_label(self) -> Dict[str, int]:
        counts_per_label_ = {label.name: 0 for label in self._labels}

        for values in self._data.values():
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                for label, count in annotation_utils.get_pixels_in_annotations(
                    wsa.annotations, labels=self._labels
                ).items():
                    if label in counts_per_label_:
                        counts_per_label_[label] += count
        return counts_per_label_

    @property
    def pixels_per_key(self):
        _counts_per_key = {}
        for file_key, values in self._data.items():
            _counts_per_key[file_key] = 0
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                _counts_per_key[file_key] += annotation_utils.get_pixels_in_annotations(
                    wsa.annotations
                )
        return _counts_per_key

    @property
    def pixels_per_label_per_key(self):
        counts_per_label_per_key_ = {}
        for file_key, values in self._data.items():
            counts_per_label_per_key_[file_key] = {}
            for wsa in values[self.__class__.ANNOTATIONS_KEY].values():
                for label, pixels in annotation_utils.get_pixels_in_annotations(
                    wsa.annotations, labels=self._labels
                ).items():
                    if label not in counts_per_label_per_key_[file_key]:
                        counts_per_label_per_key_[file_key][label] = 0
                    counts_per_label_per_key_[file_key][label] += pixels
        return counts_per_label_per_key_
