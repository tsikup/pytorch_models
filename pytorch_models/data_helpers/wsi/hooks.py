import numpy as np
from typing import List
from shapely import geometry
from wholeslidedata.annotation.hooks import AnnotationHook
from wholeslidedata.annotation.structures import Annotation


# class MaskedTiledAnnotationHook(AnnotationHook):
#
#     def __init__(self, tile_size, label_names, ratio=1, overlap=0, full_coverage=True):
#         self._tile_size = tile_size * ratio
#         self._overlap = overlap * ratio
#         self._full_coverage = full_coverage
#         self._label_names = label_names
#
#     def __call__(self, annotations: List[Annotation]):
#         new_annotations = []
#         index = 0
#         for annotation in annotations:
#             if annotation.label.name not in self._label_names:
#                 annotation._index = index
#                 new_annotations.append(annotation)
#                 index += 1
#                 continue
#
#             x1, y1, x2, y2 = annotation.bounds
#             for x in range(x1, x2, self._tile_size - self._overlap):
#                 for y in range(y1, y2, self._tile_size - self._overlap):
#                     box_poly = geometry.box(x, y, x + self._tile_size, y + self._tile_size)
#                     # Add annotation if there is intersection of box_poly with parent annotation.
#                     intersection = np.array(box_poly.intersection(annotation).exterior.coords)
#                     if intersection.size != 0:
#                         new_annotations.append(Annotation.create(
#                             index=index,
#                             type=annotation.type,
#                             coordinates=box_poly.exterior.coords,
#                             label=annotation.label.todict(),
#                         ))
#
#                         # Add intersection of box_poly with parent annotation.
#                         new_annotations[-1].mask_coordinates = intersection
#
#                         index += 1
#
#         return new_annotations


class MaskedTiledAnnotationHook(AnnotationHook):
    def __init__(
        self,
        tile_size,
        label_names,
        ratio=1,
        overlap=0,
        intersection_percentage=0.2,
        full_coverage=False,
    ):
        self._tile_size = tile_size * ratio
        self._overlap = overlap * ratio
        self._full_coverage = full_coverage
        self._label_names = label_names
        self._intersection_percentage = intersection_percentage

    def __call__(self, annotations: List[Annotation]):
        new_annotations = []
        index = 0
        for annotation in annotations:
            if annotation.label.name not in self._label_names:
                annotation._index = index
                new_annotations.append(annotation)
                index += 1
                continue

            x1, y1, x2, y2 = annotation.bounds
            for x in range(x1, x2, self._tile_size - self._overlap):
                for y in range(y1, y2, self._tile_size - self._overlap):
                    box_poly = geometry.box(
                        x, y, x + self._tile_size, y + self._tile_size
                    )
                    intersection_percentage = (
                        box_poly.intersection(annotation).area / box_poly.area
                    )
                    if (
                        not self._full_coverage
                        or intersection_percentage >= self._intersection_percentage
                    ):
                        # if not self._full_coverage or box_poly.within(annotation):
                        new_annotations.append(
                            Annotation.create(
                                index=index,
                                type=annotation.type,
                                coordinates=box_poly.exterior.coords,
                                label=annotation.label.todict(),
                            )
                        )
                        # Add intersection of box_poly with parent annotation.
                        intersections = []
                        if (
                            box_poly.intersection(annotation).type
                            == "GeometryCollection"
                        ):
                            for intersection in box_poly.intersection(annotation).geoms:
                                if intersection.type == "Polygon":
                                    intersections.append(
                                        np.array(intersection.exterior.coords)
                                    )
                                elif intersection.type == "MultiPolygon":
                                    for poly in intersection.geoms:
                                        intersections.append(
                                            np.array(poly.exterior.coords)
                                        )
                                elif intersection.type == "LineString":
                                    pass
                        else:
                            intersection = box_poly.intersection(annotation)
                            if intersection.type == "Polygon":
                                intersections.append(
                                    np.array(intersection.exterior.coords)
                                )
                            elif intersection.type == "MultiPolygon":
                                for poly in intersection.geoms:
                                    intersections.append(np.array(poly.exterior.coords))
                            else:
                                pass
                        new_annotations[-1].mask_coordinates = intersections
                        index += 1

        return new_annotations
