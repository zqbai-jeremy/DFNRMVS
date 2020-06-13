import numpy as np
import torch
from scipy.interpolate.ndgriddata import griddata
from numpy.linalg import inv as np_matinv

''' Matrices -----------------------------------------------------------------------------------------------------------
'''

def M_mat(K, R):
    """
    M = K * R, 3x3 matrix
    :param K: Camera intrinsic matrix (3x3)
    :param R: Rotation matrix (3x3)
    :return: K * R
    """
    return np.dot(K, R)


def P_mat(K, T):
    """
    Camera matrix of P = K * [R|t], 3x4 matrix
    :param K: Camera intrinsic matrix (3x3)
    :param T: Camera extrinsic matrix (3x4)
    :return: P = K * T
    """
    return np.dot(K, T)

def K_from_intrinsic(intrinsic):
    """
    Generate K matrix from intrinsic array
    :param intrinsic: array with 4 items
    :return: matrix K with 3x3 elements
    """
    return np.asarray([intrinsic[0], 0, intrinsic[2],
                       0, intrinsic[1], intrinsic[3], 0, 0, 1], dtype=np.float32).reshape(3, 3)

def scale_K(K_mat, rescale_factor):
    K = K_mat.copy()
    K *= rescale_factor
    K[2, 2] = 1.0
    return K

''' Camera Information -------------------------------------------------------------------------------------------------
'''
def Rt(T):
    """
    Return the rotation matrix and the translation vector
    :param T: transform matrix with dim (3, 4) or (4, 4)
    :return: R, t
    """
    return T[:3, :3], T[:3, 3]

def camera_center_from_Tcw(Rcw, tcw):
    """
    Compute the camera center from extrinsic matrix (world -> camera)
    :param R: Rotation matrix
    :param t: translation vector
    :return: camera center in 3D
    """
    # C = -Rcw' * t
    C = -np.dot(Rcw.transpose(), tcw)
    return C

def translation_from_center(R, C):
    """
    convert center to translation vector, C = -R^T * t -> t = -RC
    :param R: rotation of the camera, dim (3, 3)
    :param C: center of the camera
    :return: t: translation vector
    """
    t = -np.dot(R, C.reshape((3, 1)))
    return t

def camera_center_from_P(P):
    """
    Compute the camera center from camera matrix P, where P = [M | -MC], M = KR, C is the center of camera. The decompose
    method can be found in Page 163. (Multi-View Geometry Second Edition)
    :param P: Camera matrix
    :return: camera center C
    """
    X = np.linalg.det(np.asarray([P[:, 1],  P[:, 2], P[:, 3]]))
    Y = -np.linalg.det(np.asarray([P[:, 0],  P[:, 2], P[:, 3]]))
    Z = np.linalg.det(np.asarray([P[:, 0],  P[:, 1], P[:, 3]]))
    T = -np.linalg.det(np.asarray([P[:, 0],  P[:, 1], P[:, 2]]))
    C = np.asarray([X, Y, Z]) / T
    return C


def camera_pose_inv(R, t):
    """
    Compute the inverse pose
    :param R: Rotation matrix with dimension of (3x3)
    :param t: translation vector with dim of (3x1)
    :return: Camera pose matrix of (3x4)
    """
    Rwc = R.transpose()
    Ow = - np.dot(Rwc, t)
    Twc = np.eye(4, dtype=np.float32)
    Twc[:3, :3] = Rwc
    Twc[:3, 3] = Ow
    return Twc[:3, :]


def fov(fx, fy, h, w):
    """
    Camera fov on x and y dimension
    :param fx: focal length on x axis
    :param fy: focal length on y axis
    :param h:  frame height
    :param w:  frame width
    :return: fov_x, fov_y
    """
    return np.rad2deg(2*np.arctan(w / (2*fx))), np.rad2deg(2*np.arctan(h / (2*fy)))


''' Camera Projection --------------------------------------------------------------------------------------------------
'''
def pi(K, X):
    """
    [TESTED]
    Project the X in camera coordinates to the image plane
    :param K: camera intrinsic matrix array (fx, fy, cx, cy)
    :param X: point position in 3D camera coordinates system, is a 2D array with dimension of (num_points, 3)
    :return: Projected 2D pixel position u and the depth X[:, 2]
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.zeros((X.shape[0], 2), dtype=np.float32)
    u[:, 0] = fx * X[:, 0] / X[:, 2] + cx
    u[:, 1] = fy * X[:, 1] / X[:, 2] + cy
    return u, X[:, 2]


def pi_inv(K, x, d):
    """
    [TESTED]
    Project the pixel in 2D image plane and the depth to the 3D point in camera coordinate
    :param x: 2d pixel position, a 2D array with dimension of (num_points, 2)
    :param d: depth at that pixel, a array with dimension of (num_points, 1)
    :param K: camera intrinsic matrix (fx, fy, cx, cy)
    :return: 3D point in camera coordinate
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = np.zeros((x.shape[0], 3), dtype=np.float32)
    X[:, 0] = d[:, 0] * (x[:, 0] - cx) / fx
    X[:, 1] = d[:, 0] * (x[:, 1] - cy) / fy
    X[:, 2] = d[:, 0]
    return X


''' Camera Transform ---------------------------------------------------------------------------------------------------
'''
def relateive_pose(R_A, t_A, R_B, t_B):
    """
    [TESTED]
    Compute the relative pose from
    :param R_A: frame A rotation matrix
    :param t_A: frame A translation vector
    :param R_B: frame B rotation matrix
    :param t_B: frame B translation vector
    :return: 3x3 rotation matrix, 3x1 translation vector that build a 3x4 matrix of T = [R,t]

    Alternative way:

    R_{AB} = R_{B} * R_{A}^{T}
    t_{AB} = R_{B} * (C_{A} - C_{B}), where the C is the center of camera.

    >>> C_A = camera_center_from_Tcw(R_A, t_A)
    >>> C_B = camera_center_from_Tcw(R_B, t_B)
    >>> R_AB = np.dot(R_B, R_A.transpose())
    >>> t_AB = np.dot(R_B, C_A - C_B)
    """

    A_Tcw = np.eye(4, dtype=np.float32)
    A_Tcw[:3, :3] = R_A
    A_Tcw[:3, 3] = t_A
    A_Twc = np_matinv(A_Tcw)

    B_Tcw = np.eye(4, dtype=np.float32)
    B_Tcw[:3, :3] = R_B
    B_Tcw[:3, 3] = t_B

    # Transformation from A to B
    T_AB = np.dot(B_Tcw, A_Twc)
    return T_AB[:3, :]


def transpose(R, t, X):
    """
    [TESTED]
    Compute transform of the 3D points
    :param R: rotation matrix in dimension of 3x3
    :param t: translation vector
    :param X: points with 3D position, a 2D array with dimension of (num_points, 3)
    :return: transformed 3D points
    """
    assert R.shape[0] == 3
    assert R.shape[1] == 3
    assert t.shape[0] == 3
    trans_X = np.dot(R, X.transpose()).transpose() + t
    return trans_X


''' Specific Math Functions --------------------------------------------------------------------------------------------
'''
def log_quat(q):
    u = q[0:1]
    v = q[1:]
    u = np.clip(u, a_min=-1.0, a_max=1.0)
    norm = np.linalg.norm(v)
    norm = np.clip(norm, a_min=1e-6, a_max=None)
    return np.arccos(u) * v / norm


def exp_quat(log_q):
    norm = np.linalg.norm(log_q)
    norm = np.clip(norm, a_min=1e-6, a_max=None)
    u = np.cos(norm)
    v = log_q * np.sin(norm) / norm
    return np.concatenate([u, v], axis=0)

''' Stereo -------------------------------------------------------------------------------------------------------------
'''
def disparity2depth(disparities, baseline, focal_length):
    return focal_length * baseline / disparities


def depth2disparity(depth, baseline, focal_length):
    return focal_length * baseline / depth


def baseline(R_A, t_A, R_B, t_B):
    """
    Compute base line between two camera
    :param R_A: frame A rotation matrix
    :param t_A: frame A translation vector
    :param R_B: frame B rotation matrix
    :param t_B: frame B translation vector
    :return: the baseline distance
    """
    C_A = camera_center_from_Tcw(R_A, t_A)
    C_B = camera_center_from_Tcw(R_B, t_B)

    # Compute L2 norm of |C_A - C_B|
    return np.linalg.norm(C_A - C_B, ord=2)

''' Utilities ----------------------------------------------------------------------------------------------------------
'''
def x_2d_coords(h, w):
    x_2d = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[y, :, 1] = y
    for x in range(0, w):
        x_2d[:, x, 0] = x
    return x_2d


def ray2depth(camera_K, ray, pre_cache_xy_sq=None):
    """
    Convert the ray distance to depth (z)
    :param camera_K:  camera intrinsic matrix
    :param ray: light ray distance
    :param pre_cache_xy_sq: pre cached normalized coordinate square
    :return: depth
    """
    h, w = ray.shape[:2]
    fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
    if not pre_cache_xy_sq:
        xy = x_2d_coords(h, w)
        x_sq = (xy[:, :, 0] - cx) ** 2
        y_sq = (xy[:, :, 1] - cy) ** 2
        xy_sq = x_sq + y_sq
    else:
        xy_sq = pre_cache_xy_sq

    factor = np.sqrt(xy_sq + fx * fx) / fx
    return ray / factor


def dense_corres_a2b(d_a, K, Ta, Tb, pre_cache_x2d=None):
    H, W = d_a.shape
    d_a = d_a.reshape((H*W, 1))

    rel_Tcw = relateive_pose(Ta[:3, :3], Ta[:3, 3], Tb[:3, :3], Tb[:3, 3])
    if pre_cache_x2d is None:
        x_a = x_2d_coords(H, W).reshape((H*W, 2))
    else:
        x_a = pre_cache_x2d.reshape((H*W, 2))

    X_3d = pi_inv(K, x_a, d_a)
    X_3d = transpose(rel_Tcw[:3, :3], rel_Tcw[:3, 3], X_3d)
    x_2d, corr_depth = pi(K, X_3d)

    return x_2d.reshape((H, W, 2)), corr_depth


def inv_dense_corres(dense_corres_a2b, pre_cache_x2d=None):

    H, W = dense_corres_a2b.shape[:2]
    if pre_cache_x2d is None:
        x_a = x_2d_coords(H, W).reshape((H, W, 2))
    else:
        x_a = pre_cache_x2d.reshape((H, W, 2))

    dense_corres_b2a = griddata(dense_corres_a2b.reshape((H*W, 2)),
                                x_a.reshape((H*W, 2)),
                                x_a.reshape((H*W, 2)), method='linear', fill_value=-1)
    return dense_corres_b2a.reshape((H, W, 2))


def mark_out_bound_pixels(dense_corr_map, depth_map):
    assert len(dense_corr_map.shape) == 3
    assert dense_corr_map.shape[2] == 2

    H, W = dense_corr_map.shape[:2]
    out_area_y = np.logical_or(dense_corr_map[:, :, 1] > H, dense_corr_map[:, :, 1] < 0)
    out_area_x = np.logical_or(dense_corr_map[:, :, 0] > W, dense_corr_map[:, :, 0] < 0)
    depth_mask = depth_map.reshape((H, W)) < 1e-5
    out_area = np.logical_or(out_area_x, out_area_y)
    out_area = np.logical_or(out_area, depth_mask)
    return out_area


def gen_overlap_mask_img(d_a, K, Ta, Tb, pre_cache_x2d=None):
    H, W = d_a.shape
    d_a = d_a.reshape((H*W, 1))

    rel_Tcw = relateive_pose(Ta[:3, :3], Ta[:3, 3], Tb[:3, :3], Tb[:3, 3])
    if pre_cache_x2d is None:
        x_a = x_2d_coords(H, W).reshape((H*W, 2))
    else:
        x_a = pre_cache_x2d.reshape((H*W, 2))

    X_3d = pi_inv(K, x_a, d_a)
    X_3d = transpose(rel_Tcw[:3, :3], rel_Tcw[:3, 3], X_3d)
    x_2d, corr_depth = pi(K, X_3d)

    x_2d = x_2d.reshape((H, W, 2))
    out_area = mark_out_bound_pixels(x_2d, d_a)

    map = np.zeros_like(out_area, dtype=np.float32)
    map[np.logical_not(out_area)] = 1.0

    return map, x_2d


def photometric_overlap(d_a, K, Ta, Tb, pre_cache_x2d=None):
    H, W = d_a.shape
    d_a = d_a.reshape((H*W, 1))

    rel_Tcw = relateive_pose(Ta[:3, :3], Ta[:3, 3], Tb[:3, :3], Tb[:3, 3])
    if pre_cache_x2d is None:
        x_a = x_2d_coords(H, W).reshape((H*W, 2))
    else:
        x_a = pre_cache_x2d.reshape((H*W, 2))

    X_3d = pi_inv(K, x_a, d_a)
    X_3d = transpose(rel_Tcw[:3, :3], rel_Tcw[:3, 3], X_3d)
    x_2d, corr_depth = pi(K, X_3d)

    x_2d = x_2d.reshape((H, W, 2))
    out_area = mark_out_bound_pixels(x_2d, d_a)
    non_zeros = np.count_nonzero(out_area)
    valid_num_pixels = float(np.sum(d_a > 1e-5))
    if valid_num_pixels > 1e-6:
        out_ratio = float(non_zeros) / valid_num_pixels
        if out_ratio > 1.0:
            out_ratio = 1.0
        return 1.0 - out_ratio
    else:
        return 0.0


def depth2scene(d, K, Rcw, tcw, pre_cache_x2d=None):
    H, W = d.shape
    d = d.reshape((H * W, 1))

    Twc = camera_pose_inv(Rcw, tcw)
    if pre_cache_x2d is None:
        x_a = x_2d_coords(H, W).reshape((H*W, 2))
    else:
        x_a = pre_cache_x2d.reshape((H*W, 2))

    X_3d = pi_inv(K, x_a, d)
    X_3d = transpose(Twc[:3, :3], Twc[:3, 3], X_3d)

    return X_3d.reshape((H, W, 3))



def wrapping(I_a, I_b, d_a, K, R, t):
    """
    Wrap image by providing depth, rotation and translation
    :param I_a:
    :param I_b:
    :param d_a:
    :param K:
    :param R:
    :param t:
    :return:
    """
    import banet_track.ba_module as module

    H, W, C = I_a.shape
    # I_a = torch.from_numpy(I_a.transpose((2, 0, 1))).cuda().view((1, C, H, W))
    I_b = torch.from_numpy(I_b.transpose((2, 0, 1))).cuda().view((1, C, H, W))
    d_a = torch.from_numpy(d_a).cuda().view((1, H*W))
    K = torch.from_numpy(K).cuda().view(1, 3, 3)
    R = torch.from_numpy(R).cuda().view(1, 3, 3)
    t = torch.from_numpy(t).cuda().view(1, 3)

    x_a = module.x_2d_coords_torch(1, H, W).view(1, H*W, 2).cuda()                         # dim: (N, H*W, 2)
    X_a_3d = module.batched_pi_inv(K, x_a,
                                   d_a.view((1, H * W, 1)))
    X_b_3d = module.batched_transpose(R, t, X_a_3d)
    x_b_2d, _ = module.batched_pi(K, X_b_3d)
    x_b_2d_out = x_b_2d.cpu().numpy()
    x_b_2d = module.batched_x_2d_normalize(H, W, x_b_2d).view(1, H, W, 2)               # (N, H, W, 2)
    wrap_img_b = module.batched_interp2d(I_b, x_b_2d)

    return wrap_img_b.cpu().numpy().transpose((0, 2, 3, 1)).reshape((H, W, C)), x_b_2d_out.reshape(H, W, 2)