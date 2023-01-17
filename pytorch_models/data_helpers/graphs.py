import torch
import numpy as np
from PIL import Image
from typing import Union, List, Callable, Optional, Any
from torchvision import transforms
from histocartography.utils.image import WIN_SIZE as HOVERNET_WIN_SIZE
from histocartography.preprocessing import (
    NucleiExtractor,  # nuclei detector
    DeepFeatureExtractor,  # feature extractor
    HoverNetDeepFeatureExtractor,
    KNNGraphBuilder,  # graph builder,
    NucleiConceptExtractor,
    VahadaneStainNormalizer,
    MacenkoStainNormalizer,
    AugmentedDeepFeatureExtractor,  # concept extraction
)


class CellGraphExtractor:
    STAIN_NORM_METHODS = {
        "macenko": MacenkoStainNormalizer,
        "vahadane": VahadaneStainNormalizer,
    }

    def __init__(
        self,
        feature_extractor: Union[str, torch.nn.Module] = "resnet34",
        stain_norm: str = "vahadane",
        normalizer: dict = None,
        augmentations: dict = None,
        k: int = 5,
        pixel_thresh: int = None,
        add_loc_feats: bool = True,
        augment_image_level=False,
    ):
        # Stain normalization should be already be applied.

        self.normalizer = normalizer
        self.stain_normalizer = None
        if stain_norm is not None:
            self.stain_normalizer = self.STAIN_NORM_METHODS[stain_norm]()

        # 1. define nuclei extractor (HoverNet)
        self.nuclei_detector = NucleiExtractor()

        # 2. define feature extractor:
        # Extract patches of 72x72 pixels around each
        # nucleus centroid, then resize to 224 to match ResNet input size.
        self.augmentations = augmentations
        if augmentations is None:
            if feature_extractor == "hovernet":
                self.feature_extractor = HoverNetDeepFeatureExtractor(
                    model_path=None,
                    patch_size=HOVERNET_WIN_SIZE[0],
                    resize_size=HOVERNET_WIN_SIZE[0],
                    normalizer={},
                )
            else:
                self.feature_extractor = DeepFeatureExtractor(
                    architecture=feature_extractor,
                    patch_size=224,
                    resize_size=None,
                    normalizer=self.normalizer,
                )
        else:
            if feature_extractor == "hovernet":
                raise NotImplementedError
            else:
                # Augmentation on the feature level (concepts and centroids will be the same for all. only the features will be different for each augmentation)
                self.augmented_feature_extractor = AugmentedDeepFeatureExtractor(
                    rotations=augmentations["rotations"],
                    flips=augmentations["flips"],
                    architecture=feature_extractor,
                    patch_size=224,
                    resize_size=None,
                    normalizer=self.normalizer,
                )
                # Augmentation on the image level (recalculate concepts and centroids for each augmentation)
                self.feature_extractor = DeepFeatureExtractor(
                    architecture=feature_extractor,
                    patch_size=224,
                    resize_size=None,
                    normalizer=self.normalizer,
                )

        # 3. define k-NN graph builder with k=5 and thresholding edges longer
        # than 50 pixels. Add image size-normalized centroids to the node features.
        self.knn_graph_builder = KNNGraphBuilder(
            k=k, thresh=pixel_thresh, add_loc_feats=add_loc_feats
        )

        # 4. define concept extractor (can take time on large images...)
        self.nuclei_concept_extractor = NucleiConceptExtractor()

        self.augmentation_transforms = None
        if augment_image_level and augmentations:
            self.augmentation_transforms = self._build_augmentations(
                rotations=augmentations["rotations"], flips=augmentations["flips"]
            )

    def _build_augmentations(
        self,
        rotations: Optional[List[int]] = None,
        flips: Optional[List[Any]] = None,
    ) -> List[Callable]:
        """Returns a list of callable augmentation functions for the given specification
        Args:
            rotations (Optional[List[int]], optional): List of rotation angles. Defaults to None.
            flips (Optional[List[Any]], optional): List of flips. Options are no rotation "n",
                horizontal flip "h" and vertical flip "v". Defaults to None.
            padding (Optional[int], optional): Number of pixels to pad before rotation. Defaults to None.
            fill_value (Optional[int], optional): Fill value of padded pixels. Defaults to 255.
        Returns:
            List[Callable]: List of callable augmentation functions
        """
        if rotations is None:
            rotations = [0]
        if flips is None:
            flips = ["n"]
        augmentaions = list()
        for angle in rotations:
            for flip in flips:
                t = [
                    transforms.Lambda(
                        lambda x, a=angle: transforms.functional.rotate(x, angle=a)
                    )
                ]
                if flip == "h":
                    t.append(
                        transforms.Lambda(lambda x: transforms.functional.hflip(x))
                    )
                if flip == "v":
                    t.append(
                        transforms.Lambda(lambda x: transforms.functional.vflip(x))
                    )
                augmentaions.append(transforms.Compose(t))
        return augmentaions

    def get_nuclei_map(self, image):
        if self.stain_normalizer is not None:
            image = self.stain_normalizer.process(image)

        # Segment nuclei
        nuclei_map, nuclei_centroid = self.nuclei_detector.process(image)
        return nuclei_map, nuclei_centroid

    def get_graph_with_features(self, image, nuclei_map, feature_extractor):
        # Resulting node features are:
        # x for encoder
        # + 2 normalized centroid features.
        features = feature_extractor.process(image, nuclei_map)

        # Construct graph
        graph = self.knn_graph_builder.process(nuclei_map, features)

        # Handcrafted features
        concepts = self.nuclei_concept_extractor.process(image, nuclei_map)
        graph.ndata["concepts"] = torch.from_numpy(concepts).to(features.device)

        return graph, features, concepts

    def process(self, image, graph_only: bool = True):
        # ************************************** #
        # Original image + augmented on features #
        # ************************************** #
        # Original image nuclei map
        original_nuclei_map, original_nuclei_centroid = self.get_nuclei_map(image)
        if not original_nuclei_map.any():
            return None

        # Graph on original image
        graph, _, _ = self.get_graph_with_features(
            image, original_nuclei_map, feature_extractor=self.feature_extractor
        )

        if self.augmentations:
            # Graph on augmented images with centroid and edge features same as the above.
            # Only features are from the augmented images. (nodes x augmentations x features)
            augmented_graph, _, _ = self.get_graph_with_features(
                image,
                original_nuclei_map,
                feature_extractor=self.augmented_feature_extractor,
            )

        # ************************************************************** #
        # Graphs on augmented images (centroids and nuclei maps as well) #
        # ************************************************************** #
        if self.augmentation_transforms is not None:
            image_augmented_graphs = []
            for transform in self.augmentation_transforms:
                augmented_image = np.array(transform(Image.fromarray(image)))
                augmented_nuclei_map, augmented_nuclei_centroid = self.get_nuclei_map(
                    augmented_image
                )
                image_augmented_graph, _, _ = self.get_graph_with_features(
                    image,
                    augmented_nuclei_map,
                    feature_extractor=self.feature_extractor,
                )
                image_augmented_graphs.append(image_augmented_graph)

        # ****** #
        # Return #
        # ****** #
        if not graph_only:
            if self.augmentation_transforms is not None:
                return dict(
                    nuclei_map=original_nuclei_map,
                    graph=graph,
                    feature_augmented_graph=augmented_graph,
                    image_augmented_graphs=image_augmented_graphs,
                )

            if self.augmentations is not None:
                return dict(
                    nuclei_map=original_nuclei_map,
                    graph=graph,
                    feature_augmented_graph=augmented_graph,
                )

            return dict(
                nuclei_map=original_nuclei_map,
                graph=graph,
            )

        return graph
