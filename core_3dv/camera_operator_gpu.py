import numpy as np
import torch
import torch.nn.functional as F
# from banet_track.ba_optimizer import batched_mat_inv
from core_3dv.mat_util import batched_mat_inv


""" Camera Utilities ---------------------------------------------------------------------------------------------------
"""
def camera_center_from_Tcw(Rcw, tcw):
    """
    Compute the camera center from extrinsic matrix (world -> camera)
    :param R: Rotation matrix
    :param t: translation vector
    :return: camera center in 3D
    """
    # C = -Rcw' * t

    keep_dim_n = False
    if Rcw.dim() == 2:
        Rcw = Rcw.unsqueeze(0)
        tcw = tcw.unsqueeze(0)
    N = Rcw.shape[0]
    Rwc = torch.transpose(Rcw, 1, 2)
    C = -torch.bmm(Rwc, tcw.view(N, 3, 1))
    C = C.view(N, 3)

    if keep_dim_n:
        C = C.squeeze(0)
    return C


def translation_from_center(R, C):
    """
    convert center to translation vector, C = -R^T * t -> t = -RC
    :param R: rotation of the camera, dim (3, 3)
    :param C: center of the camera
    :return: t: translation vector
    """
    keep_dim_n = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
        C = C.unsqueeze(0)
    N = R.shape[0]
    t = -torch.bmm(R, C.view(N, 3, 1))
    t = t.view(N, 3)

    if keep_dim_n:
        t = t.squeeze(0)
    return t


def camera_pose_inv(R, t):
    """
    Compute the inverse pose
    :param R: rotation matrix, dim (N, 3, 3) or (3, 3)
    :param t: translation vector, dim (N, 3) or (3)
    :return: inverse pose of [R, t]
    """
    keep_dim_n = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)

    N = R.size(0)
    Rwc = torch.transpose(R, 1, 2)
    tw = -torch.bmm(Rwc, t.view(N, 3, 1))

    if keep_dim_n:
        Rwc = Rwc.squeeze(0)
        tw = tw.squeeze(0)

    return Rwc, tw


def transpose(R, t, X):
    """
    Pytorch batch version of computing transform of the 3D points
    :param R: rotation matrix in dimension of (N, 3, 3) or (3, 3)
    :param t: translation vector could be (N, 3, 1) or (3, 1)
    :param X: points with 3D position, a 2D array with dimension of (N, num_points, 3) or (num_points, 3)
    :return: transformed 3D points
    """
    keep_dim_n = False
    keep_dim_hw = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    if X.dim() == 2:
        X = X.unsqueeze(0)

    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    N = R.shape[0]
    M = X.shape[1]
    X_after_R = torch.bmm(R, torch.transpose(X, 1, 2))
    X_after_R = torch.transpose(X_after_R, 1, 2)
    trans_X = X_after_R + t.view(N, 1, 3).expand(N, M, 3)

    if keep_dim_hw:
        trans_X = trans_X.view(N, H, W, 3)
    if keep_dim_n:
        trans_X = trans_X.squeeze(0)

    return trans_X


def transform_mat44(R, t):
    """
    Concatenate the 3x4 mat [R, t] to 4x4 mat [[R, t], [0, 0, 0, 1]].
    :param R: rotation matrix, dim (N, 3, 3) or (3, 3)
    :param t: translation vector, dim (N, 3) or (3)
    :return: identical transformation matrix with dim 4x4
    """
    keep_dim_n = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)

    N = R.shape[0]
    bot = torch.tensor([0, 0, 0, 1], dtype=torch.float).to(R.device).view((1, 1, 4)).expand(N, 1, 4)
    b = torch.cat([R, t.view(N, 3, 1)], dim=2)
    out_mat44 = torch.cat([b, bot], dim=1)
    if keep_dim_n:
        out_mat44 = out_mat44.squeeze(0)

    return out_mat44


def Rt(T):
    """
    Return the rotation matrix and the translation vector
    :param T: transform matrix with dim (N, 3, 4) or (N, 4, 4), 'N' can be ignored, dim (3, 4) or (4, 4) is acceptable
    :return: R, t
    """
    if T.dim() == 2:
        return T[:3, :3], T[:3, 3]
    elif T.dim() == 3:
        return T[:, :3, :3], T[:, :3, 3]
    else:
        raise Exception("The dim of input T should be either (N, 3, 3) or (3, 3)")


def relative_pose(R_A, t_A, R_B, t_B):
    """
    Computing the relative pose from
    :param R_A: frame A rotation matrix
    :param t_A: frame A translation vector
    :param R_B: frame B rotation matrix
    :param t_B: frame B translation vector
    :return: Nx3x3 rotation matrix, Nx3x1 translation vector that build a Nx3x4 matrix of T = [R,t]
    """
    keep_dim_n = False
    if R_A.dim() == 2 and t_A.dim() == 2:
        keep_dim_n = True
        R_A = R_A.unsqueeze(0)
        t_A = t_A.unsqueeze(0)
    if R_B.dim() == 2 and t_B.dim() == 2:
        R_B = R_B.unsqueeze(0)
        t_B = t_B.unsqueeze(0)

    N = R_A.shape[0]
    A_Tcw = transform_mat44(R_A, t_A)
    A_Twc = batched_mat_inv(A_Tcw)
    B_Tcw = transform_mat44(R_B, t_B)

    # Transformation from A to B
    T_AB = torch.bmm(B_Tcw, A_Twc)
    T_AB = T_AB[:, :3, :]

    if keep_dim_n is True:
        T_AB = T_AB.squeeze(0)

    return T_AB


""" Projections --------------------------------------------------------------------------------------------------------
"""
def pi(K, X):
    """
    Projecting the X in camera coordinates to the image plane
    :param K: camera intrinsic matrix tensor (N, 3, 3) or (3, 3)
    :param X: point position in 3D camera coordinates system, is a 3D array with dimension of (N, num_points, 3), or (num_points, 3)
    :return: N projected 2D pixel position u (N, num_points, 2) and the depth X (N, num_points, 1)
    """
    keep_dim_n = False
    keep_dim_hw = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if X.dim() == 2:
        X = X.unsqueeze(0)      # make dim (1, num_points, 3)
    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    assert K.size(0) == X.size(0)
    N = K.shape[0]

    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    u_x = fx * X[:, :, 0:1] / X[:, :, 2:3] + cx
    u_y = fy * X[:, :, 1:2] / X[:, :, 2:3] + cy
    u = torch.cat([u_x, u_y], dim=-1)
    d = X[:, :, 2:3]

    if keep_dim_hw:
        u = u.view(N, H, W, 2)
        d = d.view(N, H, W)
    if keep_dim_n:
        u = u.squeeze(0)
        d = d.squeeze(0)

    return u, d


def pi_inv(K, x, d):
    """
    Projecting the pixel in 2D image plane and the depth to the 3D point in camera coordinate.
    :param x: 2d pixel position, a 2D array with dimension of (N, num_points, 2)
    :param d: depth at that pixel, a array with dimension of (N, num_points, 1)
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :return: 3D point in camera coordinate (N, num_points, 3)
    """
    keep_dim_n = False
    keep_dim_hw = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if x.dim() == 2:
        x = x.unsqueeze(0)      # make dim (1, num_points, 3)
    if d.dim() == 2:
        d = d.unsqueeze(0)      # make dim (1, num_points, 1)

    if x.dim() == 4:
        assert x.size(0) == d.size(0)
        assert x.size(1) == d.size(1)
        assert x.size(2) == d.size(2)
        assert x.size(3) == 2
        keep_dim_hw = True
        N, H, W = x.shape[:3]
        x = x.view(N, H*W, 2)
        d = d.view(N, H*W, 1)

    N = K.shape[0]
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    X_x = d * (x[:, :, 0:1] - cx) / fx
    X_y = d * (x[:, :, 1:2] - cy) / fy
    X_z = d
    X = torch.cat([X_x, X_y, X_z], dim=-1)

    if keep_dim_hw:
        X = X.view(N, H, W, 3)
    if keep_dim_n:
        X = X.squeeze(0)

    return X


def x_2d_coords(h, w, n=None):
    N = 1 if n is None else n
    x_2d = np.zeros((N, h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[:, y, :, 1] = y
    for x in range(0, w):
        x_2d[:, :, x, 0] = x
    x_2d = torch.Tensor(x_2d)
    if n is None:
        x_2d = x_2d.squeeze(0)

    return x_2d


def x_2d_normalize(h, w, x_2d):
    """
    Convert the x_2d coordinates to (-1, 1)
    :param x_2d: coordinates mapping, (N, H * W, 2)
    :return: x_2d: coordinates mapping, (N, H * W, 2), with the range from (-1, 1)
    """
    if x_2d.dim() == 4:
        N, H, W, C = x_2d.shape
        x_2d = x_2d.view(N, H*W, C)
    x_2d[:, :, 0] = (x_2d[:, :, 0] / (float(w) - 1.0))
    x_2d[:, :, 1] = (x_2d[:, :, 1] / (float(h) - 1.0))
    x_2d = x_2d * 2.0 - 1.0
    return x_2d


def dense_corres_a2b(d_a, K, Ta, Tb, pre_cache_x2d=None):
    """
    Compute dense correspondence map from a to b.
    :param d_a: depth map of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :return:
    """
    keep_dim_n = False
    if d_a.dim() == 2:
        keep_dim_n = True
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        K = K.unsqueeze(0)
    if Tb.dim() == 2:
        Tb = Tb.unsqueeze(0)

    N, H, W = d_a.shape
    d_a = d_a.view((N, H*W, 1))

    rel_Tcw = relative_pose(Ta[:, :3, :3], Ta[:, :3, 3], Tb[:, :3, :3], Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(H, W, N).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_3d = pi_inv(K, x_a_2d, d_a)
    X_3d = transpose(rel_Tcw[:, :3, :3], rel_Tcw[:, :3, 3], X_3d)
    x_2d, corr_depth = pi(K, X_3d)
    x_2d = x_2d.view((N, H, W, 2))

    if keep_dim_n is True:
        x_2d = x_2d.squeeze(0)
        corr_depth = corr_depth.squeeze(0)

    return x_2d, corr_depth


def inv_dense_corres(dense_corres_a2b, pre_cache_x2d=None):
    raise Exception('NO IMPLEMENTATION')


def mark_out_bound_pixels(dense_corr_map, depth_map):
    """
    Mark out the out of boundary correspondence
    :param dense_corr_map: dense correspondence map, dim (N, H, W, 2) or dim (H, W, 2)
    :param depth_map: depth map, dim (N, H, W), (N, H*W), (N, H*W, 1) or dim (H, W)
    :return: 'out_area': the boolean 2d array indicates correspondence that is out of boundary, dim (N, H, W) or (H, W)
    """
    keep_dim_n = False
    if dense_corr_map.dim() == 3:
        keep_dim_n = True
        dense_corr_map = dense_corr_map.unsqueeze(0)
        depth_map = depth_map.unsqueeze(0)

    N, H, W = dense_corr_map.shape[:3]
    out_area_y = (dense_corr_map[:, :, :, 1] > H) | (dense_corr_map[:, :, :, 1] < 0)
    out_area_x = (dense_corr_map[:, :, :, 0] > W) | (dense_corr_map[:, :, :, 0] < 0)

    depth_mask = depth_map.view((N, H, W)) < 1e-5
    out_area = out_area_x | out_area_y
    out_area = out_area | depth_mask

    if keep_dim_n:
        out_area = out_area.squeeze(0)

    return out_area


def gen_overlap_mask_img(d_a, K, Ta, Tb, pre_cache_x2d=None):
    """
    Generate overlap mask of project a onto b
    :param d_a: depth map of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :return: 'map':overlap mask; 'x_2d': correspondence
    """
    keep_dim_n = False
    if d_a.dim() == 2:
        keep_dim_n = True
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        K = K.unsqueeze(0)
    if Tb.dim() == 2:
        Tb = Tb.unsqueeze(0)

    N, H, W = d_a.shape
    d_a = d_a.view((N, H*W))

    rel_Tcw = relative_pose(Ta[:, :3, :3], Ta[:, :3, 3], Tb[:, :3, :3], Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(H, W, N).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_3d = pi_inv(K, x_a_2d, d_a.view((N, H*W, 1)))
    X_3d = transpose(rel_Tcw[:, :3, :3], rel_Tcw[:, :3, 3], X_3d)
    x_2d, corr_depth = pi(K, X_3d)

    x_2d = x_2d.view((N, H, W, 2))
    out_area = mark_out_bound_pixels(x_2d, d_a.view((N, H*W)))

    zeros = torch.zeros(out_area.size(), dtype=torch.float)
    ones = torch.ones(out_area.size(), dtype=torch.float)
    map = torch.where(out_area, zeros, ones)

    if keep_dim_n:
        map = map.squeeze(0)
        x_2d = x_2d.squeeze(0)

    return map, x_2d


def photometric_overlap(d_a, K, Ta, Tb, pre_cache_x2d=None):
    """
    Compute overlap ratio of project a onto b
    :param d_a: depth map of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :return: overlap ratio, dim (N)
    """
    keep_dim_n = False
    if d_a.dim() == 2:
        keep_dim_n = True
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        K = K.unsqueeze(0)
    if Tb.dim() == 2:
        Tb = Tb.unsqueeze(0)

    N, H, W = d_a.shape
    d_a = d_a.view((N, H*W))

    rel_Tcw = relative_pose(Ta[:, :3, :3], Ta[:, :3, 3], Tb[:, :3, :3], Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(H, W, N).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_3d = pi_inv(K, x_a_2d, d_a.view((N, H*W, 1)))
    X_3d = transpose(rel_Tcw[:, :3, :3], rel_Tcw[:, :3, 3], X_3d)
    x_2d, corr_depth = pi(K, X_3d)

    x_2d = x_2d.view((N, H, W, 2))
    out_area = mark_out_bound_pixels(x_2d, d_a.view(N, H, W))

    non_zeros = torch.sum(out_area.view(N, -1), dim=1).float()
    total_valid_pixels = torch.sum(d_a > 1e-5, dim=1).float()

    ones = torch.ones_like(total_valid_pixels)
    out_ratio = torch.where(total_valid_pixels < 1e-6, ones, non_zeros / total_valid_pixels)
    in_ratio = torch.clamp(1 - out_ratio, 0.0, 1.0)

    if keep_dim_n is True:
        in_ratio = in_ratio.item()

    return in_ratio


def interp2d(tensor, x_2d):
    """
    Interpolate the tensor, it will sample the pixel in input tensor by given the new coordinate (x, y) that indicates
    the position in original image.
    :param tensor: input tensor to be interpolated to a new tensor, (N, C, H, W)
    :param x_2d: new coordinates mapping, (N, H, W, 2) in (-1, 1), if out the range, it will be fill with zero
    :return: interpolated tensor
    """
    return F.grid_sample(tensor, x_2d)


def depth2scene(d, K, Rcw, tcw, pre_cache_x2d=None, in_chw_order=False):
    keep_dim_n = False
    if d.dim() == 2:
        keep_dim_n = True
        d = d.unsqueeze(0)
        Rcw = Rcw.unsqueeze(0)
        tcw = tcw.unsqueeze(0)

    N, H, W = d.shape
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(N, H, W).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    Twc = camera_pose_inv(Rcw, tcw)
    Rwc, twc = Twc[:, :3, :3], Twc[:, :3, 3]

    X_a_3d = pi_inv(K, x_a_2d, d.view((N, H * W, 1)))
    X_w_3d = transpose(Rwc, twc, X_a_3d)

    X_w_3d = X_w_3d.view(N, H, W, 3)
    if in_chw_order is True:
        X_w_3d = X_w_3d.permute(0, 3, 1, 2)

    if keep_dim_n:
        X_w_3d = X_w_3d.squeeze(0)

    return X_w_3d


def wrapping(I_b, d_a, K, Ta, Tb, pre_cache_x2d=None, in_chw_order=False):
    """
    Wrapping image from b to a
    :param I_b: image of frame b, dim (N, H, W, C) if 'in_chw_order=False' or dim (H, W, C)
    :param d_a: depth of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :param in_chw_order: indicates the format order of 'I_b', either 'chw' if 'in_chw_order=False' or hwc
    :return: wrapped image from b to a, dim is identical to 'I_b'
    """
    keep_dim_n = False
    if I_b.dim() == 3:
        keep_dim_n = True
        I_b = I_b.unsqueeze(0)
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        Tb = Tb.unsqueeze(0)
        K = K.unsqueeze(0)
    if in_chw_order is False:
        I_b = I_b.permute(0, 3, 1, 2)

    N, C, H, W = I_b.shape
    d_a = d_a.view(N, H*W)

    rel_Tcw = relative_pose(Ta[:, :3, :3], Ta[:, :3, 3], Tb[:, :3, :3], Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(N, H, W).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_a_3d = pi_inv(K, x_a_2d, d_a.view((N, H * W, 1)))
    X_b_3d = transpose(rel_Tcw[:, :3, :3], rel_Tcw[:, :3, 3], X_a_3d)
    x_b_2d, _ = pi(K, X_b_3d)

    wrap_img_b = interp2d(I_b, x_2d_normalize(H, W, x_b_2d).view((N, H, W, 2)))                       # (N, H, W, 2)
    x_b_2d = x_b_2d.view((N, H, W, 2))

    if in_chw_order is False:
        wrap_img_b = wrap_img_b.permute(0, 2, 3, 1)

    if keep_dim_n:
        wrap_img_b = wrap_img_b.squeeze(0)
        x_b_2d = x_b_2d.squeeze(0)

    return wrap_img_b, x_b_2d