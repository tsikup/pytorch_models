# import math
import numpy as np


# def rotate_point(point, origin, angle):
#     """
#     Rotate a point counterclockwise by a given angle around a given origin.
#
#     The angle should be given in radians.
#     """
#     ox, oy = origin
#     px, py = point
#
#     qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
#     qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
#     return qx, qy


def rotate_point(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def flip_point_h(point, width):
    x, y = point
    new_x = width - x - 1
    new_y = y
    return new_x, new_y


def flip_point_v(point, height):
    x, y = point
    new_y = height - y - 1
    new_x = x
    return new_x, new_y
