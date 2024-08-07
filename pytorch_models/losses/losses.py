import torch
from torch import nn
from pytorch_toolbelt import losses as L
from balanced_loss import Loss as BalancedLoss
from typing import List

from pytorch_models.losses.fair import DemographicParityLoss
from pytorch_models.losses.utils import JointLoss


# TODO: Implement other losses
# https://github.com/BloodAxe/pytorch-toolbelt/tree/develop/pytorch_toolbelt/losses
# https://github.com/JunMa11/SegLoss
# https://github.com/JunMa11/SegWithDistMap
def get_loss(
    config_losses,
    n_classes: int = None,
    classes_loss_weights: List[int] = None,
    multi_loss_weights: List[int] = None,
    samples_per_cls: List[int] = None,
    reduction: str = "mean",
):
    """
    Function to get training loss

    Parameters
    ----------
    config_losses: List of losses.
    classes_loss_weights: List of weights for each class.
    multi_loss_weights: List of weights for each loss.
    samples_per_cls: List of number of samples for each class.

    Returns
    -------
    Loss function
    """
    # if classes_loss_weights is None:
    #     classes_loss_weights = [1 for _ in config_losses]
    losses = []
    for loss in config_losses:
        if loss in ["ce", "crossentropy", "categorical_crossentropy"]:
            losses.append(
                nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(classes_loss_weights)
                    if classes_loss_weights is not None
                    else None,
                    ignore_index=-100,
                    label_smoothing=0.0,
                    reduction=reduction,
                )
            )
        elif loss in ["bce", "binary_crossentropy"]:
            losses.append(
                nn.BCEWithLogitsLoss(
                    pos_weight=torch.FloatTensor(classes_loss_weights)
                    if classes_loss_weights is not None
                    else None,
                    reduction=reduction,
                )
            )
        elif loss == "soft_ce":
            losses.append(
                L.SoftCrossEntropyLoss(
                    ignore_index=-100,
                    smooth_factor=0.0,
                    reduction=reduction,
                )
            )
        elif loss == "soft_bce":
            losses.append(
                L.SoftBCEWithLogitsLoss(
                    ignore_index=-100,
                    smooth_factor=None,
                    reduction=reduction,
                )
            )
        elif loss == "batch_balanced_bce":
            losses.append(
                L.BalancedBCEWithLogitsLoss(
                    reduction=reduction,
                )
            )
        elif loss == "binary_dice":
            # TODO: classes that contribute to loss computation and smooth
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for DiceLoss currently."
                )
            losses.append(L.DiceLoss(mode="binary", smooth=0.0))
        elif loss == "multiclass_dice":
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for DiceLoss currently."
                )
            losses.append(L.DiceLoss(mode="multiclass", smooth=0.0))
        elif loss == "binary_focal":
            losses.append(
                L.BinaryFocalLoss(
                    alpha=0.0, gamma=2.0, normalized=False, reduction=reduction
                )
            )
        elif loss == "focal":
            losses.append(
                L.FocalLoss(
                    alpha=0.0,
                    gamma=2.0,
                    normalized=False,
                    reduction=reduction,
                )
            )
        elif loss == "focal_cosine":
            losses.append(
                L.FocalCosineLoss(
                    reduction=reduction,
                )
            )
        elif loss == "binary_jaccard":
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for JaccardLoss currently."
                )
            losses.append(L.JaccardLoss(mode="binary", smooth=0.0))
        elif loss == "multiclass_jaccard":
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for JaccardLoss currently."
                )
            losses.append(L.JaccardLoss(mode="multiclass", smooth=0.0))
        elif loss == "binary_lovasz":
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for LovaszLoss currently."
                )
            losses.append(L.BinaryLovaszLoss())
        elif loss == "lovasz":
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for LovaszLoss currently."
                )
            losses.append(L.LovaszLoss())
        elif loss == "wing":
            losses.append(
                L.WingLoss(
                    reduction=reduction,
                )
            )
        elif loss == "balanced_ce":
            assert samples_per_cls is not None
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for BalancedLoss currently."
                )
            losses.append(
                BalancedLoss(
                    loss_type="cross_entropy",
                    samples_per_class=samples_per_cls,
                    class_balanced=True,
                )
            )
        elif loss == "balanced_bce":
            assert samples_per_cls is not None
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for BalancedLoss currently."
                )
            losses.append(
                BalancedLoss(
                    loss_type="binary_cross_entropy",
                    samples_per_class=samples_per_cls,
                    class_balanced=True,
                )
            )
        elif loss == "balanced_focal":
            assert samples_per_cls is not None
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for BalancedLoss currently."
                )
            losses.append(
                BalancedLoss(
                    loss_type="focal_loss",
                    samples_per_class=samples_per_cls,
                    class_balanced=True,
                )
            )
        elif loss in ["demographic_parity", "demo_parity"]:
            _n_classes = n_classes if n_classes > 2 else 2
            sensitive_classes = list(range(_n_classes))
            if reduction != "mean":
                raise NotImplementedError(
                    "Only mean reduction is supported for DemographicParityLoss currently."
                )
            losses.append(DemographicParityLoss(sensitive_classes=sensitive_classes))
        else:
            raise RuntimeError("No loss with that name.")

    if len(losses) > 1:
        return JointLoss(losses, multi_loss_weights)
    else:
        return losses[0]
