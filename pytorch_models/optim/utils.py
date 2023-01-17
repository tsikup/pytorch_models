import math


def get_warmup_factor(global_step: int, warmup_period: int, mode="linear"):
    """
    Function to get learning rate warmup factor.

    Parameters
    ----------
    global_step: current training step

    warmup_period: how many steps until warmup is finished

    mode: 'linear' or 'exp' warmup

    Returns
    -------
    A scale factor to multiply learning rate.
    """
    if mode == "exp":
        return 1.0 - math.exp(-float(global_step + 1) / warmup_period)
    elif mode == "linear":
        return min(1.0, float(global_step + 1) / warmup_period)
