import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as ttqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pytorch_models.utils.utils import setall


def calculate_mean_and_std(channels_sum, channels_squared_sum, num_batches):
    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    return mean, std


def get_channels_sums_from_ndarray(
    data: np.ndarray, channels_last=False, max_value=255.0
):
    if channels_last:
        channels = (0, 1)
    else:
        channels = (1, 2)

    if data.dtype == np.uint8:
        data = data.astype(np.float32)
        data = data / max_value
    # Mean over batch, height and width, but not over the channels
    channels_sums = data.mean(axis=channels)
    channels_squared_sum = (data**2).mean(axis=channels)
    return channels_sums, channels_squared_sum


def get_channels_sums(
    dataloader: DataLoader,
    channels_last=False,
    num_modalities=1,
    use_tqdm: bool = False,
    telegram_id=None,
    telegram_token=None,
    max_value=255.0,
    desc=None,
):
    if channels_last:
        channels = [0, 1, 2]
    else:
        channels = [0, 2, 3]

    channels_sum, channels_squared_sum, num_batches = {}, {}, {}
    setall(channels_sum, list(range(num_modalities)), 0)
    setall(channels_squared_sum, list(range(num_modalities)), 0)
    setall(num_batches, list(range(num_modalities)), 0)

    if use_tqdm and telegram_id is not None and telegram_token is not None:
        iterator = ttqdm(
            dataloader, desc=desc, token=telegram_token, chat_id=telegram_id
        )
    elif use_tqdm:
        iterator = tqdm(dataloader, desc=desc)
    else:
        iterator = dataloader

    for _data in iterator:
        if num_modalities == 1:
            _data = [_data]
        for idx, data in enumerate(_data):
            if data.dtype == torch.uint8:
                data = data.to(torch.float32)
                data = data / max_value
            # Mean over batch, height and width, but not over the channels
            channels_sum[idx] += torch.mean(data, dim=channels).detach().cpu().numpy()
            channels_squared_sum[idx] += (
                torch.mean(data**2, dim=channels).detach().cpu().numpy()
            )
            num_batches[idx] += 1

    return (channels_sum, channels_squared_sum, num_batches)


def get_mean_and_std(dataloader: DataLoader, channels_last=False):
    channels_sum, channels_squared_sum, num_batches = get_channels_sums(
        dataloader, channels_last
    )

    mean, std = calculate_mean_and_std(channels_sum, channels_squared_sum, num_batches)
    return mean, std
