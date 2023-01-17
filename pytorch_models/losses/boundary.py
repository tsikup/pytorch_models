"""
Boundary loss implementation copied from https://github.com/LIVIAETS/boundary-loss
"""
from functools import partial
from torchvision import transforms
import torch
import numpy as np
import torch.sparse
from operator import itemgetter
from torch import einsum
from torch import Tensor
from typing import Callable, Iterable, List, Set, Tuple, cast
from scipy.ndimage import distance_transform_edt as eucl_distance


class BoundaryLoss:
    """
    used as:

    class SliceSet(Dataset):
        def __init__(self):
            self.filenames: list[str]  # You get the list as you would usually
            self.dataset_root: Path  # Path to the root of the data

            self.disttransform = dist_map_transform([1, 1], 2)

        def __get__(self, n: int) -> dict[str, Tensor]:
            filename: str = self.filenames[index]

            image = Image.open(self.dataset_root / "img" / filename)
            label = Image.open(self.dataset_root / "gt" / filename)

            image_tensor: Tensor  # usual transform for an image
            one_hot_tensor: Tensor  # Usualy transform from png to one_hot encoding
            dist_map_tensor: Tensor = self.disttransform(label)

            return {"images": image_tensor,
                    "gt": one_hot_tensor,
                    "dist_map": dist_map_tensor}

    dice_loss = GeneralizedDiceLoss(idc=[0, 1])  # add here the extra params for the losses
    boundary_loss = BoundaryLoss(idc=[1])

    α = 0.01
    for data in loader:
        image: Tensor = data["images"]
        target: Tensor = data["gt"]
        dist_map_label: list[Tensor] = data["distmap"]

        pred_logits: Tensor = net(image)
        pred_probs: Tensor = F.softmax(pred_logits, dim=1)

        gdl_loss = dice_loss(pred_probs, target)
        bl_loss = boundary_loss(pred_probs, dist_map_label)  # Notice we do not give the same input to that loss
        total_loss = gdl_loss + α * bl_loss

        loss.backward()
        optimizer.step()
    """

    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        # SHOULD NOT BE ONE-HOT ENCODED!!!
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def one_hot2dist(
    seg: np.ndarray, resolution: Tuple[float, float, float] = None, dtype=None
) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = (
                eucl_distance(negmask, sampling=resolution) * negmask
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
            )
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(
        1, seg[:, None, ...], 1
    )

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose(
        [
            lambda img: np.array(img)[...],
            lambda nd: torch.tensor(nd, dtype=torch.int64)[
                None, ...
            ],  # Add one dimension to simulate batch
            partial(class2one_hot, K=K),
            itemgetter(0),  # Then pop the element to go back to img shape
        ]
    )


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose(
        [
            gt_transform(resolution, K),
            lambda t: t.cpu().numpy(),
            partial(one_hot2dist, resolution=resolution),
            lambda nd: torch.tensor(nd, dtype=torch.float32),
        ]
    )


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
