import torch
import numpy as np
from torch import nn
from pytorch_toolbelt import losses as L

from .utils import JointLoss


# TODO: Implement other losses
# https://github.com/BloodAxe/pytorch-toolbelt/tree/develop/pytorch_toolbelt/losses
# https://github.com/JunMa11/SegLoss
# https://github.com/JunMa11/SegWithDistMap
def get_loss(config_losses, classes_loss_weights=None, multi_loss_weights=None):
    """
    Function to get training loss

    Parameters
    ----------
    config_losses: List of losses.
    classes_loss_weights: List of weights for each class.
    multi_loss_weights: List of weights for each loss.

    Returns
    -------
    Loss function
    """
    if classes_loss_weights is None:
        classes_loss_weights = [1 for _ in config_losses]
    losses = []
    for loss in config_losses:
        if loss in ["ce", "crossentropy", "categorical_crossentropy"]:
            losses.append(
                nn.CrossEntropyLoss(
                    # weight=torch.FloatTensor(classes_loss_weights) if classes_loss_weights is not None else None,
                    ignore_index=-100,
                    label_smoothing=0.0,
                )
            )
        elif loss in ["bce", "binary_crossentropy"]:
            losses.append(nn.BCEWithLogitsLoss())
        elif loss == "soft_ce":
            losses.append(L.SoftCrossEntropyLoss(ignore_index=-100, smooth_factor=0.0))
        elif loss == "soft_bce":
            losses.append(
                L.SoftBCEWithLogitsLoss(ignore_index=-100, smooth_factor=None)
            )
        elif loss == "balanced_bce":
            losses.append(L.BalancedBCEWithLogitsLoss())
        elif loss == "binary_dice":
            # TODO: classes that contribute to loss computation and smooth
            losses.append(L.DiceLoss(mode="binary", smooth=0.0))
        elif loss == "multiclass_dice":
            losses.append(L.DiceLoss(mode="multiclass", smooth=0.0))
        elif loss == "binary_focal":
            losses.append(L.BinaryFocalLoss(alpha=0.0, gamma=2.0, normalized=False))
        elif loss == "focal":
            losses.append(L.FocalLoss(alpha=0.0, gamma=2.0, normalized=False))
        elif loss == "focal_cosine":
            losses.append(L.FocalCosineLoss())
        elif loss == "binary_jaccard":
            losses.append(L.JaccardLoss(mode="binary", smooth=0.0))
        elif loss == "multiclass_jaccard":
            losses.append(L.JaccardLoss(mode="multiclass", smooth=0.0))
        elif loss == "binary_lovasz":
            losses.append(L.BinaryLovaszLoss())
        elif loss == "lovasz":
            losses.append(L.LovaszLoss())
        elif loss == "wing":
            losses.append(L.WingLoss())
        else:
            raise RuntimeError("No loss with that name.")

    if len(losses) > 1:
        return JointLoss(losses, multi_loss_weights)
    else:
        return losses[0]
