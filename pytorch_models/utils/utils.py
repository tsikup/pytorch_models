def to_tuple(x, shape=3):
    if isinstance(x, (tuple, list)):
        return list(x)
    else:
        return [
            x,
        ] * shape


def setall(d, keys, value):
    for k in keys:
        d[k] = value
