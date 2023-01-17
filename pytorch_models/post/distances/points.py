import numpy as np
import torch
from torch import Tensor


def cosine_weight(
    center: [Tensor, np.ndarray, tuple, list],
    point: [Tensor, np.ndarray, tuple, list],
    mode="pt",
):
    """
    As described in: Meng, Z., Zhao, Z., Li, B., Su, F., Guo, L. and Wang, H., 2020. Triple up-sampling segmentation
    network with distribution consistency loss for pathological diagnosis of cervical precancerous lesions. IEEE
    Journal of Biomedical and Health Informatics, 25(7), pp.2673-2685.

    Wcos(xij) = cos(π / 2 · √((i − ic)^2 + (j − jc)^2) / √(ic^2 + jc^2))

    Where i,j is the point for which we want to find the distance weight from the center point ic, jc
    """
    if mode == "pt":
        return torch.cos(
            torch.pi
            / 2
            * torch.linalg.norm(torch.abs(center - point))
            / torch.linalg.norm(center)
        )
    elif mode == "np_meshgrid":
        Xc, Yc = center
        X, Y = point
        return np.cos(
            np.pi
            / 2
            * np.sqrt((Xc - X) ** 2 + (Yc - Y) ** 2)
            / np.sqrt(Xc**2 + Yc**2)
        )
        # return np.sqrt(X**2 + Y**2)
    elif mode == "np":
        Xc, Yc = center
        X, Y = point
        result = np.zeros(X.shape + Y.shape)
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                result[i, j] = np.cos(
                    np.pi
                    / 2
                    * np.sqrt((Xc - x) ** 2 + (Yc - y) ** 2)
                    / np.sqrt(Xc**2 + Yc**2)
                )
        return result


def exp_weight(
    center: [Tensor, np.ndarray, tuple, list],
    point: [Tensor, np.ndarray, tuple, list],
    mode="pt",
):
    """
    As described in: Meng, Z., Zhao, Z., Li, B., Su, F., Guo, L. and Wang, H., 2020. Triple up-sampling segmentation
    network with distribution consistency loss for pathological diagnosis of cervical precancerous lesions. IEEE
    Journal of Biomedical and Health Informatics, 25(7), pp.2673-2685.

    Wexp(xij) = exp(- √((i − ic)^2 + (j − jc)^2) / √(ic^2 + jc^2))

    Where i,j is the point for which we want to find the distance weight from the center point ic, jc
    """
    if mode == "pt":
        return torch.exp(
            -1
            * torch.linalg.norm(torch.abs(center - point))
            / torch.linalg.norm(center)
        )
    elif mode == "np_meshgrid":
        Xc, Yc = center
        X, Y = point
        return np.exp(
            -1 * np.sqrt((Xc - X) ** 2 + (Yc - Y) ** 2) / np.sqrt(Xc**2 + Yc**2)
        )
        # return np.sqrt(X**2 + Y**2)


def radial_base_weight(
    center: [Tensor, np.ndarray, tuple, list],
    point: [Tensor, np.ndarray, tuple, list],
    mode="pt",
):
    """
    As described in: Meng, Z., Zhao, Z., Li, B., Su, F., Guo, L. and Wang, H., 2020. Triple up-sampling segmentation
    network with distribution consistency loss for pathological diagnosis of cervical precancerous lesions. IEEE
    Journal of Biomedical and Health Informatics, 25(7), pp.2673-2685.

    Wrbf(xij) = exp(- ((i − ic)^2 + (j − jc)^2) / 2ic^2)

    Where i,j is the point for which we want to find the distance weight from the center point ic, jc. Also the image
    must be square, and thus ic == jc.
    """
    # image must be a square and for its center pixel Xc == Yc
    if mode == "pt":
        Xc, Yc = center
        assert Xc == Yc
        X, Y = point
        return torch.exp(
            -1 * (torch.pow(X - Xc, 2) + torch.pow(Y - Yc, 2)) / (2 * torch.pow(Xc, 2))
        )
    elif mode == "np_meshgrid":
        Xc, Yc = center
        X, Y = point
        return np.exp(-1 * ((X - Xc) ** 2 + (Y - Yc) ** 2) / (2 * Xc**2))
