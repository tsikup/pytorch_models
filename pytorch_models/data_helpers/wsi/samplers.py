import copy
from typing import Union, Dict, List
import numpy as np
from wholeslidedata import WholeSlideAnnotation
from wholeslidedata.samplers.annotationsampler import AnnotationSampler
from wholeslidedata.samplers.batchreferencesampler import BatchReferenceSampler
from wholeslidedata.samplers.labelsampler import LabelSampler
from wholeslidedata.samplers.patchlabelsampler import PatchLabelSampler
from wholeslidedata.samplers.samplesampler import SampleSampler
from wholeslidedata.samplers.structures import BatchShape

from .wholeslideimage import MultiResWholeSlideImage
from .files import MultiResWholeSlideImageFile


class MultiResPatchSampler:
    def __init__(
        self,
        center=True,
        relative=False,
        tissue_percentage=0.5,
        blurriness_threshold=500,
    ):
        self._center = center
        self._relative = relative
        self.tissue_percentage = tissue_percentage
        self.blurriness_threshold = blurriness_threshold

    def sample(
        self,
        image: Union[MultiResWholeSlideImage, MultiResWholeSlideImageFile],
        point,
        size,
        pixel_spacings,
    ):
        assert isinstance(
            pixel_spacings, dict
        ), "pixel_spacings should be a dictionary with the spacings for 'context', 'target', 'graph' and 'details'"
        image_opened = isinstance(image, MultiResWholeSlideImage)

        if not image_opened:
            wsi = image.open()
        else:
            wsi = image

        patch = wsi.get_data(
            point.x,
            point.y,
            *size,
            pixel_spacings,
            center=self._center,
            relative=self._relative,
            tissue_percentage=self.tissue_percentage,
            blurriness_threshold=self.blurriness_threshold
        )

        ratio = dict()
        for key, value in pixel_spacings.items():
            ratio[key] = wsi.get_downsampling_from_spacing(value)

        if not image_opened:
            wsi.close()
            wsi = None
            del wsi

        return patch, ratio


class MultiResSampleSampler(SampleSampler):
    def __init__(
        self,
        patch_sampler: MultiResPatchSampler,
        patch_label_sampler: PatchLabelSampler,
        batch_shape: BatchShape,
        sample_callbacks=None,
    ):
        self._batch_shape = batch_shape
        self._patch_sampler = patch_sampler
        self._patch_label_sampler = patch_label_sampler
        self._sample_callbacks = sample_callbacks

    def sample(self, wsi: MultiResWholeSlideImage, wsa: WholeSlideAnnotation, point):
        x_samples = self._init_samples()
        y_samples = self._init_samples()

        spacings = dict()
        names = []
        for key in x_samples:
            name = key[0]
            names.append(name)
            spacing = key[1]
            spacings[name] = spacing

        assert (
            "target" in names
        ), "Your BatchShape instance should contain 'target' shape and spacing."

        patch_shape = list(x_samples[("target", spacings["target"])].keys())[0]

        x_sample, y_sample = self._sample(point, wsi, wsa, patch_shape, spacings)

        for key, value in spacings.items():
            if key in list(x_sample.keys()):
                x_samples[(key, value)][tuple(patch_shape)] = x_sample[key]
            if key in list(y_sample.keys()):
                y_samples[(key, value)][tuple(patch_shape)] = y_sample[key]

        self._reset_sample_callbacks()
        return x_samples, y_samples

    def _sample(
        self,
        point: tuple,
        wsi: MultiResWholeSlideImage,
        wsa: WholeSlideAnnotation,
        patch_shape: Union[tuple, list],
        pixel_spacings: Dict[str, float],
    ):

        data, ratio = self._patch_sampler.sample(
            wsi, point, patch_shape[:2], pixel_spacings
        )

        label = dict()
        for key in pixel_spacings:
            if key == "graph":
                continue
            label[key] = self._patch_label_sampler.sample(
                wsa=wsa,
                point=point,
                size=patch_shape[:2],
                ratio=ratio[key],
            )

        for key in pixel_spacings:
            if key != "graph":
                data[key], label = self._apply_sample_callbacks(data[key], label)

        del_values = []
        for key in data:
            if not key in pixel_spacings:
                del_values.append(key)
        for key in del_values:
            del data[key]
        return data, label

    def _init_samples(self):
        return {
            tuple(_spacing): {tuple(input_size): [] for input_size in sizes}
            for _spacing, sizes in self._batch_shape.items()
        }

    def _apply_sample_callbacks(self, patch, mask):
        if self._sample_callbacks:
            for callback in self._sample_callbacks:
                patch, mask = callback(patch, mask)

        return patch, mask


@LabelSampler.register(("ordered_onetime",))
class OrderedLabelOneTimeSampler(LabelSampler):
    def __init__(self, annotations_per_label: dict[str, int], seed: int = 123):
        labels = [
            [
                label,
            ]
            * counts
            for label, counts in annotations_per_label.items()
        ]
        labels = [label for sublist in labels for label in sublist]
        super().__init__(labels, seed=seed)
        self._labels_cycle = iter(self._labels)
        self.reset()

    def __len__(self):
        return len(self._labels)

    def __next__(self):
        try:
            return next(self._labels_cycle)
        except StopIteration:
            return None

    def reset(self):
        self._labels_cycle = iter(self._labels)

    def update(self, batch):
        pass


class BatchOneTimeReferenceSampler(BatchReferenceSampler):
    def __init__(
        self, dataset, batch_size, label_sampler, annotation_sampler, point_sampler
    ):

        super(BatchOneTimeReferenceSampler, self).__init__(
            dataset, batch_size, label_sampler, annotation_sampler, point_sampler
        )

    def __len__(self):
        try:
            return len(self._label_sampler)
        except AttributeError:
            return None

    def batch(self):
        batch = []
        for _ in range(self._batch_size):
            # get next label
            label = next(self._label_sampler)

            if label is None:
                continue

            # get next index of label
            index = next(self._annotation_sampler)(label)

            # get new sample to samples
            sample = self._dataset.sample_references[label][index]

            point = self._point_sampler.sample(sample)

            # add new sample to samples
            batch.append({"reference": sample, "point": point})

        return batch
