import numpy as np
import torch
from torchvision.utils import make_grid
import os
from PIL import Image
import cv2

# External libs
from external.face3d.face3d import mesh
from external.face3d.face3d.morphable_model.load import load_BFM_info

# Internal libs
import data.BFM.utils as bfm_utils


def visualize_geometry(vert, back_ground, tri, face_region_mask=None, gt_flag=False):
    """
    Visualize untextured mesh
    :param vert: mesh vertices. np.array: (nver, 3)
    :param back_ground: back ground image. np.array: (256, 256, 3)
    :param tri: mesh triangles. np.array: (ntri, 3) int32
    :param face_region_mask: mask for valid vertices. np.array: (nver, 1) bool
    :param gt_flag: Whether render with ESRC ground truth mesh. The normals of BFM (predicted mesh) point to the
                    opposite direction, thus need to multiply by -1.
    :return: image_t: rendered image. np.array: (3, 256, 256)
    """
    colors = np.ones((vert.shape[0], 3), dtype=np.float) - 0.25
    if gt_flag:
        sh_coeff = np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    else:
        sh_coeff = np.array((0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    colors = mesh.light.add_light_sh(vert, tri, colors, sh_coeff)
    projected_vertices = vert.copy()  # using stantard camera & orth projection
    h = w = 256
    c = 3
    image_vert = mesh.transform.to_image(projected_vertices, h, w)
    if face_region_mask is not None:
        image_vert, colors, tri = bfm_utils.filter_non_tight_face_vert(image_vert, colors, tri, face_region_mask)
    image_t = mesh.render.render_colors(image_vert, tri, colors, h, w, BG=back_ground)
    image_t = np.minimum(np.maximum(image_t, 0), 1).transpose((2, 0, 1))
    return image_t


def visualization(verts, img, bfm, n_sample=None, MM_base_dir='./external/face3d/examples/Data/BFM'):
    bfm_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
    face_region_mask = bfm.face_region_mask.copy()
    face_region_mask[bfm_info['nose_hole'].ravel()] = False
    N, V, _, _, _ = img.shape
    img_grids = []
    for i in range(N):
        if n_sample is not None and i >= n_sample:
            break
        img_list = []
        for j in range(V):
            cur_img = img[i, j, ...].cpu()
            cur_img_np = np.ascontiguousarray(cur_img.numpy().transpose((1, 2, 0)))
            img_list.append(cur_img)
            for k in range(len(verts)):
                if k == 0:
                    vert = verts[k][i, j, ...].detach().cpu().numpy()
                else:
                    vert = verts[k][-1][i, j, ...].detach().cpu().numpy()

                geo_vis = visualize_geometry(vert, np.copy(cur_img_np), bfm.model['tri'], face_region_mask)
                img_list.append(torch.tensor(geo_vis))
        img_grid = make_grid(img_list, nrow=1 + len(verts)).detach().cpu()
        img_grids.append(img_grid)

    return img_grids


def correct_landmark_verts(verts, bfm, bfm_torch):
    N, V, nver, _ = verts[0].shape

    # Get landmark and neighbor idx
    kpt_neib_idx = bfm.neib_vert_idx[bfm.kpt_ind, :]            # (68, max_number_neighbor_per_vert)
    kpt_neib_idx = kpt_neib_idx[kpt_neib_idx < nver]
    # kpt_idx = np.concatenate([bfm.kpt_ind, kpt_neib_idx], axis=0)

    for k in range(1, len(verts)):
        vert = verts[k][-1]

        # Compute laplacian mean filtered vertices
        vert = vert.view(N * V, nver, 3)
        vert_t = torch.cat([vert, torch.zeros_like(vert[:, :1, :])], dim=1)  # (N * V, nver + 1, 3)
        vert_neib = vert_t[:, bfm.neib_vert_idx.ravel(), :].view(N * V, nver, bfm.neib_vert_idx.shape[1], 3)
        vert_neib_sum = torch.sum(vert_neib, dim=2)  # (N * V, nver, 3)
        vert_lapla_mean = vert_neib_sum / bfm_torch.neib_vert_count.view(1, nver, 1).float()

        # Replace lamdmark vertices with laplacian mean
        vert[:, bfm.kpt_ind, :] = 0.9 * vert_lapla_mean[:, bfm.kpt_ind, :] + 0.1 * vert[:, bfm.kpt_ind, :]

        # # Compute laplacian mean filtered vertices
        # vert = vert.view(N * V, nver, 3)
        # vert_t = torch.cat([vert, torch.zeros_like(vert[:, :1, :])], dim=1)  # (N * V, nver + 1, 3)
        # vert_neib = vert_t[:, bfm.neib_vert_idx.ravel(), :].view(N * V, nver, bfm.neib_vert_idx.shape[1], 3)
        # vert_neib_sum = torch.sum(vert_neib, dim=2)  # (N * V, nver, 3)
        # vert_lapla_mean = vert_neib_sum / bfm_torch.neib_vert_count.view(1, nver, 1).float()
        #
        # # Replace lamdmark vertices with laplacian mean
        # vert[:, kpt_neib_idx, :] = vert_lapla_mean[:, kpt_neib_idx, :]

        verts[k][-1] = vert.view(N, V, nver, 3)

    return verts


def load_img_2_tensors(image_path, fa, face_detector, transform_func=None):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(
        img,
        top=50,
        bottom=50,
        left=50,
        right=50,
        borderType=cv2.BORDER_DEFAULT
    )
    s = 1.5e3
    t = [0, 0, 0]
    scale = 1.2
    size = 256
    ds = face_detector.detect_from_image(img[..., ::-1].copy())
    for i in range(len(ds)):
        d = ds[i]
        center = [d[3] - (d[3] - d[1]) / 2.0, d[2] - (d[2] - d[0]) / 2.0]
        center[0] += (d[3] - d[1]) * 0.06
        center[0] = int(center[0])
        center[1] = int(center[1])
        l = max(d[2] - d[0], d[3] - d[1]) * scale
        if l < 200:
            continue
        x_s = center[1] - int(l / 2)
        y_s = center[0] - int(l / 2)
        x_e = center[1] + int(l / 2)
        y_e = center[0] + int(l / 2)
        t = [256. - center[1] + t[0], center[0] - 256. + t[1], 0]
        rescale = size / (x_e - x_s)
        s *= rescale
        t = [t[0] * rescale, t[1] * rescale, 0.]
        img = Image.fromarray(img).crop((x_s, y_s, x_e, y_e))
        img = cv2.resize(np.asarray(img), (size, size)).astype(np.float32)
        break
    assert img.shape[0] == img.shape[1] == 256
    ori_img_tensor = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32) / 255.0)  # (C, H, W)
    img_tensor = ori_img_tensor.clone()
    if transform_func:
        img_tensor = transform_func(img_tensor)

    # Get 2D landmarks on image
    kpts_list = fa.get_landmarks(img)
    kpts = kpts_list[0]
    kpts_tensor = torch.from_numpy(kpts)                                                    # (68, 2)

    return img_tensor, ori_img_tensor, kpts_tensor
