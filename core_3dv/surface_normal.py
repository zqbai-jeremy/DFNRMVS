import numpy as np
import cv2


def compute_surface_normal(depth_map):
    # Reference:
    # http://answers.opencv.org/question/82453/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-product/
    #
    h = depth_map.shape[0]
    w = depth_map.shape[1]

    f1 = np.asarray([1,  2,  1, 0,  0,  0, -1, -2, -1], dtype=np.float32).reshape((3, 3)) / 8.0
    f2 = np.asarray([1, 0, -1, 2, 0, -2, 1, 0, -1], dtype=np.float32).reshape((3, 3)) / 8.0
    f1m = cv2.flip(f1, 0)
    f2m = cv2.flip(f2, 1)

    n1 = -1.0 * cv2.filter2D(depth_map, -1, f1m, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_CONSTANT)
    n2 = -1.0 * cv2.filter2D(depth_map, -1, f2m, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_CONSTANT)
    n3 = 1.0 / np.sqrt(n1 * n1 + n2 * n2 + 1.0)

    surface_normal = np.zeros((h, w, 3), dtype=np.float32)
    surface_normal[:, :, 1] = n1 * n3
    surface_normal[:, :, 0] = n2 * n3
    surface_normal[:, :, 2] = n3
    # Has been normalized, no need further processing
    return surface_normal


def azimuth_map(surface_normal):
    h = surface_normal.shape[0]
    w = surface_normal.shape[1]
    proj_norm = surface_normal[:, :, :2]
    x_axis = np.asarray((1.0, 0.0), dtype=np.float32)
    az_map = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h):
        for x in range(0, w):
            normal = proj_norm[y, x]
            az_map[y, x] = np.arccos(np.dot(normal, x_axis) / (np.linalg.norm(normal)))
    return np.rad2deg(az_map)
