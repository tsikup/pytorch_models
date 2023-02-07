import json
import warnings
import numpy as np
from typing import List
from wholeslidedata.annotation.types import Annotation
from wholeslidedata.annotation.labels import Label, Labels
from wholeslidedata.annotation.parser import AnnotationParser


class QuPathAnnotationParser(AnnotationParser):
    @staticmethod
    def get_available_labels(opened_annotation: dict):
        labels = []
        for annotation in opened_annotation:
            try:
                _label = Label.create(
                    annotation["properties"]["classification"]["name"],
                )
                labels.append(_label)
            except Exception:
                pass

        return Labels.create(set(labels))

    def _parse(self, path) -> List[dict]:
        with open(path) as json_file:
            data = json.load(json_file)
        if type(data) is not list:
            data = [data]
        labels = self._get_labels(data)
        for annotation in data:
            try:
                label_name = annotation["properties"]["classification"]["name"].lower()
            except:
                label_name = None
            if label_name not in labels.names:
                continue
            label = labels.get_label_by_name(label_name)

            if "label" not in annotation:
                annotation["label"] = dict()

            for key, value in label.todict().items():
                if key not in annotation["label"] or annotation["label"][key] is None:
                    annotation["label"][key] = value

            yield annotation

    def parse(self, path, spacing=None) -> List[Annotation]:

        if not self._path_exists(path):
            raise FileNotFoundError(path)

        if self._empty_file(path):
            warn = f"Loading empty file: {path}"
            warnings.warn(warn)
            return []

        annotations = []
        for index, annotation in enumerate(self._parse(path)):
            annotation["index"] = index
            annotation["type"] = annotation["geometry"]["type"].lower()

            assert annotation["type"] in [
                "polygon",
                "multipolygon",
            ], "Annotation type should be polygon or multipolygon."

            if annotation["type"] == "polygon":
                annotation["coordinates"] = np.array(
                    annotation["geometry"]["coordinates"][0]
                )
                annotation["label"] = self._rename_label(annotation["label"])
                if spacing is not None:
                    annotation["spacing"] = spacing
                del annotation["properties"]
                del annotation["geometry"]
                annotation.update(self._kwargs)
                annotations.append(Annotation.create(**annotation))
            elif annotation["type"] == "multipolygon":
                for coords in annotation["geometry"]["coordinates"]:
                    new_annotation = dict(
                        index=index,
                        type="polygon",
                        coordinates=coords[0],
                        label=self._rename_label(annotation["label"]),
                    )
                    if spacing is not None:
                        new_annotation["spacing"] = spacing
                    new_annotation.update(self._kwargs)
                    annotations.append(Annotation.create(**new_annotation))

        for callback in self._callbacks:
            annotations = callback(annotations)
        return annotations
