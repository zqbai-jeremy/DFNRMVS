import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as plt_path
import scipy.io as sio
import os
import functools
import copy
import json
from numba import jit
from array import array

# External libs
from external.face3d.face3d import mesh
import external.face3d.face3d.morphable_model.load as bfm_load
from external.face3d.face3d.morphable_model import MorphabelModel
from external.face3d.face3d.morphable_model.load import load_BFM_info
from external.face3d.face3d.mesh.light import get_normal
import face_alignment.detection.sfd as face_detector_module

# Internal libs
import core_3dv.camera_operator_gpu as cam_opt_gpu


# Functions ------------------------------------------------------------------------------------------------------------
def load_3DMM(MM_base_dir='./external/face3d/examples/Data/BFM'):
    """
    Load BFM 3DMM.
    additional attr:
        params_mean_3dffa:
        params_std_3dffa:
        sigma_exp:
        face_region_mask: whether a vertex is in the face region. np.array boolean. (nver, 1)
        tri_idx: index of adjacent triangles for each vertex. np.array int32. (nver, max_number_tri_per_vert)
        updated expPC & expEV: 64 parameters.
        uv_coords: y axis point to up, range [0, 1]. np.array (nver, 2)
        uv_coords_64: uv_coords mapped to 64x64 image space with zero depths. np.array (nver, 3)
        pixel_vert_idx_64: vertices indices of rendered triangle on each pixel in a 64x64 uv map. np.array int. (64, 64, 3)
        pixel_vert_weights_64: vertices weights of rendered triangle on each pixel in a 64x64 uv map. np.array (64, 64, 3)
    :param MM_base_dir:
    :return:
    """
    def LoadExpBasis():
        n_vertex = 53215
        Expbin = open(os.path.join(MM_base_dir, 'Out/Exp_Pca.bin'), 'rb')
        exp_dim = array('i')
        exp_dim.fromfile(Expbin, 1)
        expMU = array('f')
        expPC = array('f')
        expMU.fromfile(Expbin, 3 * n_vertex)
        expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)
        expPC = np.array(expPC)
        expPC = np.reshape(expPC, [exp_dim[0], -1])
        expPC = np.transpose(expPC)
        expEV = np.loadtxt(os.path.join(MM_base_dir, 'Out/std_exp.txt')).reshape((exp_dim[0], -1))
        expPC = expPC[:, :64].astype(np.float32)
        expEV = expEV[:64, :].astype(np.float32)
        return expPC, expEV

    def process_uv(uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
        return uv_coords

    bfm = MorphabelModel(os.path.join(MM_base_dir, 'Out/BFM.mat'))
    model_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
    bfm.segbin = model_info['segbin'].astype(np.bool)

    # params_attr = sio.loadmat(os.path.join(MM_base_dir, 'Out/params_attr.mat'))
    # bfm.params_mean_3dffa = params_attr['params_mean']
    # bfm.params_std_3dffa = params_attr['params_std']
    # sigma_exp = sio.loadmat(os.path.join(MM_base_dir, 'Out/sigma_exp.mat'))
    # bfm.sigma_exp = sigma_exp['sigma_exp'].reshape((29, 1))

    bfm.face_region_mask = get_tight_face_region(bfm, MM_base_dir, True)

    bfm.tri_idx = get_adjacent_triangle_idx(int(bfm.nver), bfm.model['tri'])
    bfm.neib_vert_idx, bfm.neib_vert_count = get_neighbor_vert_idx(int(bfm.nver), bfm.tri_idx, bfm.model['tri'])

    bfm.model['expPC'], bfm.model['expEV'] = LoadExpBasis()
    bfm.n_exp_para = 64
    bfm.model['shapePC'] = bfm.model['shapePC'][:, :80]
    bfm.model['shapeEV'] = bfm.model['shapeEV'][:80, :]
    bfm.n_shape_para = 80

    bfm.uv_coords = bfm_load.load_uv_coords(os.path.join(MM_base_dir, 'Out/BFM_UV.mat'))
    bfm.uv_coords[:, 1] -= 0.11
    bfm.uv_coords[:, 0] = (bfm.uv_coords[:, 0] - 0.5) * 12./11. + 0.5
    bfm.uv_coords[:, 1] = (bfm.uv_coords[:, 1] - 0.5) * 4./3. + 0.5
    bfm.uv_coords_32 = process_uv(bfm.uv_coords.copy(), 32, 32)
    bfm.pixel_vert_idx_32, bfm.pixel_vert_weights_32 = \
        get_pixel_vert_idx_and_weights(bfm.uv_coords_32, bfm.model['tri'], 32, 32)
    bfm.uv_coords_64 = process_uv(bfm.uv_coords.copy(), 64, 64)
    bfm.pixel_vert_idx_64, bfm.pixel_vert_weights_64 = \
        get_pixel_vert_idx_and_weights(bfm.uv_coords_64, bfm.model['tri'], 64, 64)
    bfm.uv_coords_128 = process_uv(bfm.uv_coords.copy(), 128, 128)
    bfm.pixel_vert_idx_128, bfm.pixel_vert_weights_128 = \
        get_pixel_vert_idx_and_weights(bfm.uv_coords_128, bfm.model['tri'], 128, 128)
    return bfm


def denormalize_BFM_params(bfm, sp, ep, tp, mode):
    """
    Denormalize BFM params
    :param bfm:
    :param sp: shape params. (n_shape_para, 1)
    :param ep: expression params. (n_exp_para, 1)
    :param tp: texture params. (n_tex_para, 1)
    :param mode: use which distribution: normal_manual, normal_3ddfa
    :return:
    """
    if mode == 'normal_manual':
        sp = sp * bfm.model['shapeEV']
        ep = ep * bfm.model['expEV']
    elif mode == 'normal_3ddfa':
        sp = sp * bfm.params_std_3dffa[7 : 7+199, :] #+ bfm.params_mean_3dffa[7 : 7+199, :]
        ep = ep * bfm.params_std_3dffa[7 + 199 :, :] #+ bfm.params_mean_3dffa[7 + 199 :, :]
        # print(bfm.params_std_3dffa[7 : 7+199, :])
        # print(bfm.params_std_3dffa[7 + 199 :, :])
    elif mode == 'normal_mvfnet':
        sp = sp * bfm.model['shapeEV']
        ep = ep * 1.0 / (1000.0 * bfm.sigma_exp)
    else:
        raise NotImplementedError('Have not implemented %s mode in denormalize().' % mode)

    tp = tp * np.sqrt(bfm.model['texEV']) * 12.5

    return sp, ep, tp


def normalize_pose_params(pitch, yaw, roll, s, tx, ty):
    return pitch / 180. * np.pi, yaw / 180. * np.pi, roll / 180. * np.pi, (s - 1.25e-03) / 5e-04, tx / 10., ty / 10.


def denormalize_pose_params(pitch, yaw, roll, s, tx, ty):
    return pitch * 180. / np.pi, yaw * 180. / np.pi, roll * 180. / np.pi, s * 5e-04 + 1.25e-03, tx * 10., ty * 10.


def add_light_sh_rgb(vertices, triangles, colors, sh_coeff, normal=None):
    '''
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    --> can be expressed in terms of spherical harmonics(omit the lighting coefficients)
    I = albedo * (sh(n) x sh_coeff)

    albedo: n x 1
    sh_coeff: 27 x 1
    Y(n) = (1, n_x, n_y, n_z, n_xn_y, n_xn_z, n_yn_z, n_x^2 - n_y^2, 3n_z^2 - 1)': n x 9
    # Y(n) = (1, n_x, n_y, n_z)': n x 4

    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3] albedo
        sh_coeff: [9, 1] spherical harmonics coefficients

    Returns:
        lit_colors: [nver, 3]
    '''
    assert vertices.shape[0] == colors.shape[0]
    nver = vertices.shape[0]
    r_sh_coeff = sh_coeff[0::3, :]
    g_sh_coeff = sh_coeff[1::3, :]
    b_sh_coeff = sh_coeff[2::3, :]
    if normal is not None:
        n = normal
    else:
        n = mesh.light.get_normal(vertices, triangles)  # [nver, 3]
    sh = np.array((np.ones(nver), n[:, 0], n[:, 1], n[:, 2], n[:, 0] * n[:, 1], n[:, 0] * n[:, 2], n[:, 1] * n[:, 2],
                   n[:, 0] ** 2 - n[:, 1] ** 2, 3 * (n[:, 2] ** 2) - 1)).T  # [nver, 9]
    r_ref = sh.dot(r_sh_coeff)  # [nver, 1]
    g_ref = sh.dot(g_sh_coeff)  # [nver, 1]
    b_ref = sh.dot(b_sh_coeff)  # [nver, 1]
    lit_colors = np.copy(colors)
    lit_colors[:, 0] = lit_colors[:, 0] * r_ref[:, 0]
    lit_colors[:, 1] = lit_colors[:, 1] * g_ref[:, 0]
    lit_colors[:, 2] = lit_colors[:, 2] * b_ref[:, 0]
    return lit_colors


def get_tight_face_region(bfm, MM_base_dir='./external/face3d/examples/Data/BFM', front=False):
    """
    Get vertices indices of tight face region
    :param bfm:
    :param MM_base_dir:
    :return: mask: True if the vertex is inside tight face region. np.array boolean. (nver, 1)
    """
    bfm_info = bfm_load.load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
    sp = np.zeros((bfm.n_shape_para, 1), dtype=np.float)
    ep = np.zeros((bfm.n_exp_para, 1), dtype=np.float)
    vertices = bfm.generate_vertices(sp, ep)

    nver = vertices.shape[0]
    nose_bottom = vertices[bfm.kpt_ind[30], :]
    nose_bridge = (vertices[bfm.kpt_ind[39], :] + vertices[bfm.kpt_ind[42], :]) / 2  # between the inner eye corners
    face_centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    outer_eye_dist = np.linalg.norm(vertices[bfm.kpt_ind[36], :] - vertices[bfm.kpt_ind[45], :])
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)

    # Mask lower face
    mask_radius = 1.5 * (outer_eye_dist + nose_dist) / 2
    # dist = np.linalg.norm(vertices[:nver, :] - face_centre.reshape((1, 3)), axis=1)
    # face_region_mask1 = dist <= mask_radius
    face_region_mask1 = (np.square((vertices[:nver, 0] - nose_bridge[0]) / (1.2 * mask_radius)) + np.square(
        (vertices[:nver, 1] - nose_bridge[1]) / (1.075 * mask_radius)) + np.square(
        (vertices[:nver, 2] - nose_bridge[2]) / (0.65 * mask_radius))) <= 1

    # Mask forehead
    in_ellipse = (np.square((vertices[:nver, 0] - nose_bridge[0]) / mask_radius) + np.square(
        (vertices[:nver, 1] - nose_bridge[1]) / (0.5 * mask_radius)) + np.square(
        (vertices[:nver, 2] - nose_bridge[2]) / (0.75 * mask_radius))) <= 1
    face_region_mask2 = np.logical_or(np.logical_and(in_ellipse, vertices[:nver, 1] > nose_bridge[1]),
                                      vertices[:nver, 1] < nose_bridge[1])
    face_region_mask = np.logical_and(face_region_mask1, face_region_mask2)
    return face_region_mask.reshape(-1, 1)

    # vertices[:, 0] = 1e5 * np.arctan(vertices[:, 0] / vertices[:, 2])
    #
    # if not front:
    #     face_contour_line_xyz = vertices[bfm_info['face_contour_line'].ravel(), :]
    #     face_contour_line_xyz = np.concatenate([face_contour_line_xyz[16:80, :],
    #                                             face_contour_line_xyz[81:208, :],
    #                                             face_contour_line_xyz[209:272, :],
    #                                             face_contour_line_xyz[344:454, :]], axis=0) # remove strange points
    # else:
    #     face_contour_line_xyz = vertices[bfm_info['face_contour_front_line'].ravel(), :]
    #     face_contour_line_xyz = np.concatenate([face_contour_line_xyz[:87, :],
    #                                             face_contour_line_xyz[88:215, :],
    #                                             face_contour_line_xyz[216:, :]], axis=0)  # remove strange points
    #
    # # for i in range(1, face_contour_line_xyz.shape[0]):
    #     # if face_contour_line_xyz[i, 1] - face_contour_line_xyz[i - 1, 1] > 5e4:
    #     # if face_contour_line_xyz[i, 0] >= 130000 or face_contour_line_xyz[i, 0] <= -130000:
    #     #     print(i)
    # # print(face_contour_front_line_xyz.shape)
    # # plt.plot(face_contour_line_xyz[:, 0], face_contour_line_xyz[:, 1])
    #
    # polygon = plt_path.Path(face_contour_line_xyz[:, :2])
    # xy_mask = polygon.contains_points(vertices[:, :2])
    # z_mask = vertices[:, 2] >= np.amin(face_contour_line_xyz[:, 2])
    # forehead_mask = vertices[:, 1] <= 6e4 - (vertices[:, 0] / 1e6)**2 * 2e6
    # mask = np.logical_and(xy_mask, z_mask)
    # mask = np.logical_and(mask, forehead_mask)
    # # mask[bfm_info['nose_hole'].ravel()] = False
    # return mask.reshape(-1, 1)


def filter_non_tight_face_vert(vertices, colors, triangles, mask):
    idx = np.arange(0, colors.shape[0], 1).astype(np.int)[mask.ravel()]
    mapping = np.array([-1] * colors.shape[0])
    mapping[idx] = np.arange(idx.shape[0])
    new_tri = np.zeros_like(triangles) - 1

    @jit(nopython=True)
    def filter_triangles(triangles, new_tri, mapping):
        n_tri = 0
        for i in range(triangles.shape[0]):
            flag = True
            for j in range(triangles.shape[1]):
                if mapping[triangles[i, j]] < 0:
                    flag = False
                    break
            if flag:
                for j in range(triangles.shape[1]):
                    new_tri[n_tri, j] = mapping[triangles[i, j]]
                n_tri += 1
        return new_tri, n_tri

    new_tri, n_tri = filter_triangles(triangles, new_tri, mapping)
    new_tri = new_tri[:n_tri, :]
    vertices = vertices[mask.ravel()]
    colors = colors[mask.ravel()]
    return vertices, colors, new_tri


def get_adjacent_triangle_idx(nver, triangles):
    """
    Compute the indices of triangles that a vertiex is belong to
    :param nver:
    :param triangles: np.array (ntri, 3)
    :return: tri_idx: np.array (nver, max_number_tri_per_vert)
    """
    ntri = triangles.shape[0]
    tri_idx = np.zeros((nver, 10), dtype=np.int32) + ntri
    tri_count = np.zeros(nver, dtype=np.int32)

    @jit(nopython=True)
    def get_tri_idx(tri, tri_idx, tri_count):
        for i in range(tri.shape[0]):
            for j in range(3):
                vert_i = tri[i, j]
                tri_count_vert = tri_count[vert_i]
                tri_idx[vert_i, tri_count_vert] = i
                tri_count[vert_i] += 1
        return tri_idx, tri_count

    tri_idx, tri_count = get_tri_idx(triangles, tri_idx, tri_count)
    tri_count_max = np.amax(tri_count)
    tri_idx = tri_idx[:, :tri_count_max]
    return tri_idx


def get_neighbor_vert_idx(nver, tri_idx, triangles):
    """
    Get the neighborhood vertices indices of each vertex
    :param nver:
    :param tri_idx: indices of triangles each vertex is belong to. np.array (nver, max_number_tri_per_vert)
    :param triangles: np.array (ntri, 3)
    :return: neib_vert_idx: np.array (nver, max_number_neighbor_per_vert)
    """
    neib_vert_idx = np.zeros((nver, 10), dtype=np.int32) + nver
    neib_vert_count = np.zeros(nver, dtype=np.int32)
    ntri = triangles.shape[0]

    @jit(nopython=True)
    def get_vert_idx(nver, ntri, tri, tri_idx, neib_vert_idx, neib_vert_count):
        for i in range(nver):
            for j in tri_idx[i, :]:
                if j == ntri:
                    continue
                for k in tri[j, :]:
                    if k == i:
                        continue
                    isin = False
                    for t in neib_vert_idx[i, :]:
                        if k == t:
                            isin = True
                            break
                    if not isin:
                        neib_vert_idx[i, neib_vert_count[i]] = k
                        neib_vert_count[i] += 1
        return neib_vert_idx, neib_vert_count

    neib_vert_idx, neib_vert_count = get_vert_idx(nver, ntri, triangles, tri_idx, neib_vert_idx, neib_vert_count)
    neib_vert_count_max = np.amax(neib_vert_count)
    neib_vert_idx = neib_vert_idx[:, :neib_vert_count_max]
    return neib_vert_idx, neib_vert_count


def get_pixel_vert_idx_and_weights(vertices, triangles, h, w):
    depth_buffer, triangle_buffer, barycentric_weight = mesh.render.rasterize_triangles(vertices, triangles, h, w)
    triangle_buffer[triangle_buffer == -1] = 0
    pixel_vert_idx = triangles[triangle_buffer, :]
    return pixel_vert_idx, barycentric_weight


def render_face_image_from_params(sp_norm, tp_norm, ep_norm, sh_coeff,
                                  pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm,
                                  bfm):
    sp, ep, tp = denormalize_BFM_params(bfm, sp_norm, ep_norm, tp_norm, mode='normal_3ddfa')
    vertices = bfm.generate_vertices(sp, ep)
    colors = bfm.generate_colors(tp)
    colors = np.minimum(np.maximum(colors, 0), 1)
    if sh_coeff.shape[0] == 9:
        colors = mesh.light.add_light_sh(vertices, bfm.model['tri'], colors, sh_coeff)
    elif sh_coeff.shape[0] == 27:
        colors = add_light_sh_rgb(vertices, bfm.model['tri'], colors, sh_coeff)
    else:
        raise NotImplemented('sh_coeff size (%d, 1) not support' % sh_coeff.shape[0])
    pitch, yaw, roll, s, tx, ty = denormalize_pose_params(pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm)
    angles = [pitch, yaw, roll]
    t = [tx, ty, 0]
    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection
    h = w = 256
    c = 3
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image_t = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w,
                                        BG=np.ones((h, w, c), dtype=np.float32))
    image_t = np.minimum(np.maximum(image_t, 0), 1)

    return image_t


def render_face_geometry_from_params(sp_norm, ep_norm, pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm,
                                     back_ground, bfm):
    tp_norm = np.zeros((bfm.n_tex_para, 1), dtype=np.float)
    sp, ep, tp = denormalize_BFM_params(bfm, sp_norm, ep_norm, tp_norm, mode='normal_manual')#'normal_3ddfa')
    vertices = bfm.generate_vertices(sp, ep)
    vertices[:, 2] -= 7.5e4
    colors = np.ones((vertices.shape[0], 3), dtype=np.float)# + 0.5
    # sh_coeff = np.array((0.7, 0.0, -0.4, 0.4, 0.0, 0.0, 0.0, 0.4, 0.4), dtype=np.float).reshape((9, 1))
    sh_coeff = np.array((0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    colors = mesh.light.add_light_sh(vertices, bfm.model['tri'], colors, sh_coeff)
    pitch, yaw, roll, s, tx, ty = denormalize_pose_params(pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm)
    angles = [pitch, yaw, roll]
    t = [tx, ty, 0]
    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection
    h = w = 256
    c = 3
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image_vertices, colors, triangles = filter_non_tight_face_vert(image_vertices, colors, bfm.triangles, bfm.face_region_mask)
    image_t = mesh.render.render_colors(image_vertices, triangles, colors, h, w, BG=back_ground)
    image_t = np.minimum(np.maximum(image_t, 0), 1)

    return image_t


def render_face_illumination_from_params(sp_norm, ep_norm, pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm,
                                         sh_coeff, back_ground, bfm):
    tp_norm = np.zeros((bfm.n_tex_para, 1), dtype=np.float)
    sp, ep, tp = denormalize_BFM_params(bfm, sp_norm, ep_norm, tp_norm, mode='normal_3ddfa')
    vertices = bfm.generate_vertices(sp, ep)
    colors = np.ones((vertices.shape[0], 3), dtype=np.float) - 0.5
    if sh_coeff.shape[0] == 9:
        colors = mesh.light.add_light_sh(vertices, bfm.model['tri'], colors, sh_coeff)
    elif sh_coeff.shape[0] == 27:
        colors = add_light_sh_rgb(vertices, bfm.model['tri'], colors, sh_coeff)
    else:
        raise NotImplemented('sh_coeff size (%d, 1) not support' % sh_coeff.shape[0])
    pitch, yaw, roll, s, tx, ty = denormalize_pose_params(pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm)
    angles = [pitch, yaw, roll]
    t = [tx, ty, 0]
    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection
    h = w = 256
    c = 3
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image_t = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w,
                                        BG=back_ground)
    image_t = np.minimum(np.maximum(image_t, 0), 1)

    return image_t


# Classes --------------------------------------------------------------------------------------------------------------
class BFMSyntheticJson(object):
    """
    BFM Synthetic data collection

    for each sample dict:
        'id': index. int
        'sp_norm': normalized identity params. np.array
        'tp_norm': normalized texture params. np.array
        'views': list of dict for images of different views
            'image_path': image file path from dataset_dir
            'ep_norm': normalized expression params. np.array
            'sh_coeff': Spherical Harmonics (SH) coefficients. np.array
            'pitch_norm': pitch (x) rotation angle in radius. float
            'yaw_norm': yaw (y) rotation angle in radius. float
            'roll_norm': roll (z) rotation angle in radius. float
            's_norm': normalized scale. float
            'tx_norm': normalized x translation on 2D image plane. float
            'ty_norm': normalized x translation on 2D image plane. float
    """

    def __init__(self, json_path=None):
        super(BFMSyntheticJson, self).__init__()
        self.samples = []
        if json_path is not None:
            self.load_json(json_path)

    def __len__(self):
        return len(self.samples)

    def sort_by_frame_idx(self):
        def idx_comparator(a, b):
            if a['id'] < b['id']:
                return -1
            elif a['id'] > b['id']:
                return 1
            else:
                return 0
        self.samples = sorted(self.samples, key=functools.cmp_to_key(idx_comparator))

    def append_to_views(self, views, image_path, ep_norm, sh_coeff,
                        pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm):
        """
        Append a new view to views
        :param views: existing views list
        :param image_path:
        :param ep_norm:
        :param sh_coeff:
        :param pitch_norm:
        :param yaw_norm:
        :param roll_norm:
        :param s_norm:
        :param tx_norm:
        :param ty_norm:
        :return:
        """
        views.append({
            'image_path': image_path,
            'ep_norm': ep_norm,
            'sh_coeff': sh_coeff,
            'pitch_norm': pitch_norm,
            'yaw_norm': yaw_norm,
            'roll_norm': roll_norm,
            's_norm': s_norm,
            'tx_norm': tx_norm,
            'ty_norm': ty_norm
        })
        return views

    def append_sample(self, sp_norm, tp_norm, views):
        """
        Add a sample to the collection
        :param sp_norm:
        :param tp_norm:
        :param views: list
        :return:
        """
        sample_dict = {}
        sample_dict['id'] = len(self.samples)
        sample_dict['sp_norm'] = sp_norm
        sample_dict['tp_norm'] = tp_norm
        sample_dict['views'] = views
        self.samples.append(sample_dict)

    def dump_to_json(self, json_path):
        self.sort_by_frame_idx()

        samples = copy.deepcopy(self.samples)
        for sample in samples:
            sample['sp_norm'] = sample['sp_norm'].ravel().tolist()
            sample['tp_norm'] = sample['tp_norm'].ravel().tolist()
            for view in sample['views']:
                view['ep_norm'] = view['ep_norm'].ravel().tolist()
                view['sh_coeff'] = view['sh_coeff'].ravel().tolist()

        with open(json_path, 'w') as out_json_file:
            json.dump(samples, out_json_file, indent=2)

    def load_json(self, json_path):
        with open(json_path, 'r') as json_file:
            json_instance = json.load(json_file)
            self.samples = json_instance
            for sample in self.samples:
                sample['sp_norm'] = np.asarray(sample['sp_norm'], dtype=np.float32).reshape((-1, 1))
                sample['tp_norm'] = np.asarray(sample['tp_norm'], dtype=np.float32).reshape((-1, 1))
                for view in sample['views']:
                    view['ep_norm'] = np.asarray(view['ep_norm'], dtype=np.float32).reshape((-1, 1))
                    view['sh_coeff'] = np.asarray(view['sh_coeff'], dtype=np.float32).reshape((-1, 1))
        self.sort_by_frame_idx()


class MorphabelModel_torch(object):
    """
    PyTorch version of external.face3d.face3d.morphable_model.MorphabelModel
    """
    def __init__(self, bfm):
        super(MorphabelModel_torch, self).__init__()
        self.nver = int(bfm.nver)
        self.ntri = bfm.ntri
        self.n_shape_para = bfm.n_shape_para
        self.n_exp_para = bfm.n_exp_para
        self.n_tex_para = bfm.n_tex_para
        self.kpt_ind = bfm.kpt_ind
        self.triangles = bfm.triangles
        self.full_triangles = bfm.full_triangles

        self.model = {}
        self.model['shapeMU'] = torch.from_numpy(bfm.model['shapeMU']).unsqueeze(0)     # (1, 3 * nver, 1)
        self.model['shapePC'] = torch.from_numpy(bfm.model['shapePC']).unsqueeze(0)     # (1, 3 * nver, n_shape_para)
        self.model['shapeEV'] = torch.from_numpy(bfm.model['shapeEV']).unsqueeze(0)     # (1, n_shape_para, 1)
        self.model['expMU'] = torch.from_numpy(bfm.model['expMU']).unsqueeze(0)         # (1, 3 * nver, 1)
        self.model['expPC'] = torch.from_numpy(bfm.model['expPC']).unsqueeze(0)         # (1, 3 * nver, n_exp_para)
        self.model['expEV'] = torch.from_numpy(bfm.model['expEV']).unsqueeze(0)         # (1, n_exp_para, 1)
        self.model['texMU'] = torch.from_numpy(bfm.model['texMU']).unsqueeze(0)         # (1, 3 * nver, 1)
        self.model['texPC'] = torch.from_numpy(bfm.model['texPC']).unsqueeze(0)         # (1, 3 * nver, n_tex_para)
        self.model['texEV'] = torch.from_numpy(bfm.model['texEV']).unsqueeze(0)         # (1, n_tex_para, 1)

        # self.params_mean_3dffa = torch.from_numpy(bfm.params_mean_3dffa.astype(np.float32)).unsqueeze(0)
        # self.params_std_3dffa = torch.from_numpy(bfm.params_std_3dffa.astype(np.float32)).unsqueeze(0)
        # self.sigma_exp = torch.from_numpy(bfm.sigma_exp.astype(np.float32)).unsqueeze(0)

        self.uv_coords_32 = torch.from_numpy(bfm.uv_coords_32.astype(np.float32)).unsqueeze(0)      # (1, nver, 3)
        self.pixel_vert_idx_32 = torch.from_numpy(bfm.pixel_vert_idx_32.astype(np.int32))           # (32, 32, 3)
        self.pixel_vert_weights_32 = \
            torch.from_numpy(bfm.pixel_vert_weights_32.astype(np.float32)).unsqueeze(0)             # (1, 32, 32, 3)

        self.uv_coords_64 = torch.from_numpy(bfm.uv_coords_64.astype(np.float32)).unsqueeze(0)      # (1, nver, 3)
        self.pixel_vert_idx_64 = torch.from_numpy(bfm.pixel_vert_idx_64.astype(np.int32))           # (64, 64, 3)
        self.pixel_vert_weights_64 = \
            torch.from_numpy(bfm.pixel_vert_weights_64.astype(np.float32)).unsqueeze(0)             # (1, 64, 64, 3)

        self.uv_coords_128 = torch.from_numpy(bfm.uv_coords_128.astype(np.float32)).unsqueeze(0)    # (1, nver, 3)
        self.pixel_vert_idx_128 = torch.from_numpy(bfm.pixel_vert_idx_128.astype(np.int32))         # (128, 128, 3)
        self.pixel_vert_weights_128 = \
            torch.from_numpy(bfm.pixel_vert_weights_128.astype(np.float32)).unsqueeze(0)            # (1, 128, 128, 3)

        self.neib_vert_count = torch.from_numpy(bfm.neib_vert_count)                                # (nver,)
        self.face_region_mask = torch.from_numpy(bfm.face_region_mask).unsqueeze(0)                 # (1, nver, 1)

    def cuda(self, device):
        for key in self.model.keys():
            self.model[key] = self.model[key].cuda(device)

        # self.params_mean_3dffa = self.params_mean_3dffa.cuda(device)
        # self.params_std_3dffa = self.params_std_3dffa.cuda(device)
        # self.sigma_exp = self.sigma_exp.cuda(device)
        self.uv_coords_32 = self.uv_coords_32.cuda(device)
        self.pixel_vert_idx_32 = self.pixel_vert_idx_32.cuda(device)
        self.pixel_vert_weights_32 = self.pixel_vert_weights_32.cuda(device)
        self.uv_coords_64 = self.uv_coords_64.cuda(device)
        self.pixel_vert_idx_64 = self.pixel_vert_idx_64.cuda(device)
        self.pixel_vert_weights_64 = self.pixel_vert_weights_64.cuda(device)
        self.neib_vert_count = self.neib_vert_count.cuda(device)
        self.face_region_mask = self.face_region_mask.cuda(device)

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (N, n_shape_para, 1)
            exp_para: (N, n_exp_para, 1)
        Returns:
            vertices: (N, nver, 3)
        '''
        N = shape_para.shape[0]
        vertices = self.model['shapeMU'].expand(N, -1, -1) + \
                   torch.bmm(self.model['shapePC'].expand(N, -1, -1), shape_para) + \
                   torch.bmm(self.model['expPC'].expand(N, -1, -1), exp_para)
        vertices = vertices.view(N, self.nver, 3)#.transpose(1, 2)
        return vertices

    def denormalize_BFM_params(self, sp=None, ep=None, tp=None, mode='normal_3ddfa'):
        """
        Denormalize BFM params
        :param bfm:
        :param sp: shape params. (N, n_shape_para, 1)
        :param ep: expression params. (N, n_exp_para, 1)
        :param tp: texture params. (N, n_tex_para, 1)
        :param mode: use which distribution: normal_manual, normal_3ddfa, normal_mvfnet
        :return:
        """
        if mode == 'normal_manual':
            sp = sp * self.model['shapeEV'] if sp is not None else None
            ep = ep * self.model['expEV'] if ep is not None else None
        # elif mode == 'normal_3ddfa':
        #     sp = sp * self.params_std_3dffa[:, 7 : 7 + self.n_shape_para, :] if sp is not None else None
        #     ep = ep * self.params_std_3dffa[:, 7 + self.n_shape_para :, :] if ep is not None else None
        # elif mode == 'normal_mvfnet':
        #     sp = sp * self.model['shapeEV'] if sp is not None else None
        #     ep = ep * 1.0 / (1000.0 * self.sigma_exp) if ep is not None else None
        else:
            raise NotImplementedError('Have not implemented %s mode in denormalize().' % mode)
        tp = tp * torch.sqrt(self.model['texEV']) * 12.5 if tp is not None else None
        return sp, ep, tp

    def angle2matrix(self, angles):
        """
        Get rotation matrix from three rotation angles(degree). right-handed.
        :param angles: Euler angle x, y, z (pitch, yaw, roll). (N, 3)
        :return: rotation matrix (N, 3, 3)
        """
        N = angles.shape[0]
        x, y, z = angles[:, 0] * np.pi / 180.0, angles[:, 1] * np.pi / 180.0, angles[:, 2] * np.pi / 180.0
        zero = torch.zeros(1, dtype=torch.float).to(angles.device)
        one = torch.ones(1, dtype=torch.float).to(angles.device)

        # x
        Rx = [torch.stack([one[0], zero[0], zero[0],
                           zero[0], torch.cos(x[i]), -torch.sin(x[i]),
                           zero[0], torch.sin(x[i]), torch.cos(x[i])]) for i in range(N)]
        Rx = torch.stack(Rx, dim=0).view(N, 3, 3)

        # y
        Ry = [torch.stack([torch.cos(y[i]), zero[0], torch.sin(y[i]),
                           zero[0], one[0], zero[0],
                           -torch.sin(y[i]), zero[0], torch.cos(y[i])]) for i in range(N)]
        Ry = torch.stack(Ry, dim=0).view(N, 3, 3)

        # z
        Rz = [torch.stack([torch.cos(z[i]), -torch.sin(z[i]), zero[0],
                           torch.sin(z[i]), torch.cos(z[i]), zero[0],
                           zero[0], zero[0], one[0]]) for i in range(N)]
        Rz = torch.stack(Rz, dim=0).view(N, 3, 3)

        R = torch.bmm(Rz, torch.bmm(Ry, Rx))
        return R

    def transform(self, vertices, s, angles, t3d):
        """
        Transform vertices based on scale, Euler angles, 3D translation
        :param vertices: (N, nver, 3)
        :param s: scale. (N, 1)
        :param angles: Euler angle x, y, z (pitch, yaw, roll). (N, 3)
        :param t3d: 3D translation. (N, 3)
        :return:
        """
        N, M, _ = vertices.shape
        scaled_vert = s.view(N, 1, 1) * vertices
        R = self.angle2matrix(angles)
        t = t3d.view(N, 3, 1)
        vert = cam_opt_gpu.transpose(R, t, scaled_vert)
        return vert

    def to_image(self, vertices, h, w):
        ''' change vertices to image coord system
        3d system: XYZ, center(0, 0, 0)
        2d image: x(u), y(v). center(w/2, h/2), flip y-axis.
        Args:
            vertices: [N, nver, 3]
            h: height of the rendering
            w : width of the rendering
        Returns:
            projected_vertices: [N, nver, 3]
        '''
        # move to center of image
        vertices_w = vertices[:, :, 0] + w / 2
        vertices_h = vertices[:, :, 1] + h / 2
        # flip vertices along y-axis.
        vertices_h = h - vertices_h - 1.0
        return torch.stack([vertices_w, vertices_h, vertices[:, :, 2]], dim=2)

    def backface_culling_cpu(self, vertices, triangles):
        """
        Get visibility mask of vertices
        :param vertices: torch.tensor. (N, nver, 3)
        :param triangles: np.array. (ntri, 3)
        :return: vis_mask: np.array boolean. (N, nver, 1)
        """
        N = vertices.shape[0]
        vertices = vertices.detach().cpu().numpy()
        normal = [get_normal(vertices[i, :, :], triangles) for i in range(N)]
        normal = np.stack(normal, axis=0)       # (N, nver, 3)
        vis_mask = normal[:, :, 2:3] < 0.0
        return vis_mask

    def backface_culling(self, vert, bfm):
        N, nver, _ = vert.shape

        # Compute normal loss
        pt0 = vert[:, bfm.model['tri'][:, 0], :]  # (N, ntri, 3)
        pt1 = vert[:, bfm.model['tri'][:, 1], :]  # (N, ntri, 3)
        pt2 = vert[:, bfm.model['tri'][:, 2], :]  # (N, ntri, 3)
        tri_normal = torch.cross(pt0 - pt1, pt0 - pt2, dim=-1)  # (N, ntri, 3). normal of each triangle
        tri_normal = torch.cat([tri_normal, torch.zeros_like(tri_normal[:, :1, :])], dim=1)  # (N, ntri + 1, 3)
        vert_tri_normal = tri_normal[:, bfm.tri_idx.ravel(), :].view(N, nver, bfm.tri_idx.shape[1], 3)
        normal = torch.sum(vert_tri_normal, dim=2)  # (N, nver, 3)

        # Compute mask
        vis_mask = torch.lt(normal[:, :, 2:3], 0.0)
        return vis_mask
