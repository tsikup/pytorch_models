import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(
        nrow, ncol, sharex="all", sharey="all", figsize=(ncol * 4, nrow * 4)
    )

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])


def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x, y: x + y, zip(*img_arrays))
    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))


def plot_errors(results_dict, title):
    markers = itertools.cycle(("+", "x", "o"))

    plt.title("{}".format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel("dice_coef")
        plt.xlabel("epoch")
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()


def masks_to_colorimg(masks):
    colors = np.asarray(
        [
            (201, 58, 64),
            (242, 207, 1),
            (0, 152, 75),
            (101, 172, 228),
            (56, 34, 132),
            (160, 194, 56),
        ]
    )

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:, y, x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y, x, :] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f"Output mask (class {i + 1})")
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f"Output mask")
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
