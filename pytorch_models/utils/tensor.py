from typing import List, Tuple, Union
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch import Tensor


def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c) or (b, a)
    :param B: has shape (b, k, p) or (b, k)
    :return: (b, ak, cp) or (b, ak)
    """
    assert (A.dim() == 3 and B.dim() == 3) or (A.dim() == 2 and B.dim() == 2)

    if A.dim() == 2:
        res = torch.einsum("ba,bk->bak", A, B).view(A.size(0), A.size(1) * B.size(1))
    elif A.dim() == 3:
        res = torch.einsum("bac,bkp->bakcp", A, B).view(
            A.size(0), A.size(1) * B.size(1), A.size(2) * B.size(2)
        )
    return res


def aggregate_features(
    features: Union[List[torch.Tensor], Tuple[torch.Tensor]], method=None
):
    if method == "concat":
        h = torch.cat(features, dim=-1)
    elif method == "average" or method == "mean":
        h = torch.stack(features, dim=-1)
        h = torch.mean(h, dim=-1)
    elif method == "max":
        h = torch.stack(features, dim=-1)
        h = torch.max(h, dim=-1)[0]
    elif method == "min":
        h = torch.stack(features, dim=-1)
        h = torch.min(h, dim=-1)[0]
    elif method == "mul":
        if len(features) == 2:
            h = torch.mul(*features)
        else:
            h = torch.stack(features, dim=-1)
            h = torch.prod(h, dim=-1)
    elif method == "add" or method == "sum":
        if len(features) == 2:
            h = torch.add(*features)
        else:
            h = torch.stack(features, dim=-1)
            h = torch.sum(h, dim=-1)
    elif method == "lse":
        h = [torch.exp(f) for f in features]
        h = torch.stack(h, dim=-1)
        h = torch.log(torch.mean(h, dim=-1, keepdim=True)).squeeze(-1)
    elif method == "kron":
        # h = torch.kron(*features)
        h = kronecker_product_einsum_batched(*features)
    else:
        h = features[0]
    return h


def tensor_to_image(tensor: torch.Tensor, keepdim: bool = False) -> "np.ndarray":
    """Converts a PyTorch tensor image to a numpy image.

    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor: image of the form :math:`(H, W)`, :math:`(C, H, W)` or
            :math:`(B, C, H, W)`.
        keepdim: If ``False`` squeeze the input image to match the shape
            :math:`(H, W, C)` or :math:`(H, W)`.

    Returns:
        image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.

    Example:
        # >>> img = torch.ones(1, 3, 3)
        # >>> tensor_to_image(img).shape
        (3, 3)

        # >>> img = torch.ones(3, 4, 4)
        # >>> tensor_to_image(img).shape
        (4, 4, 3)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: "np.ndarray" = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"Cannot process tensor with shape {input_shape}")

    return image


def pad_tensor(tensor, size, value=255):
    """
    Pad image to (size,size,-1)

    Args:
      tensor: Image as NumPy array.
      size: Size to pad image to.
      value: constant value to pad with.
    """

    width = torch.tensor(tensor.shape[1])
    height = torch.tensor(tensor.shape[2])
    padding_x = size - width
    assert padding_x >= 0
    padding_y = size - height
    assert padding_y >= 0

    if padding_y == 0 and padding_x == 0:
        return tensor

    padding_x1 = torch.floor(padding_x / 2).int()
    padding_x2 = torch.ceil(padding_x / 2).int()

    padding_y1 = torch.floor(padding_y / 2).int()
    padding_y2 = torch.ceil(padding_y / 2).int()

    padded_img = F.pad(
        tensor,
        padding=(padding_x1, padding_y1, padding_x2, padding_y2),
        fill=value,
        padding_mode="constant",
    )

    return padded_img


class UnNormalize(object):
    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)
        self.unnormalize = T.Normalize(
            (-self.mean / self.std).tolist(), (1.0 / self.std).tolist()
        )

    def __call__(self, tensor: Tensor, channels_last: bool = False):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: UnNormalized image.
        """
        assert len(tensor.shape) < 4, "Cannot accept batched or 4D data yet."
        assert tensor.shape[0] <= 3, "Channels should be at first dim."
        _tensor = self.unnormalize.forward(tensor)
        if channels_last:
            return _tensor.permute(1, 2, 0)
        return _tensor
