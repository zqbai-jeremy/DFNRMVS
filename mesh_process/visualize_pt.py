import numpy as np

from core_3dv.camera_operator import x_2d_coords, pi_inv, transpose
from visualizer.visualizer_3d import Visualizer
from data.utils import get_eyes_nose_3D_landmarks


def save_rendered_res_preds(file_name, img, depth, K, Twc, preds, preds_depth, R, t):
    with open(file_name, 'wb') as f:
        np.savez(f, img=img, depth=depth, K=K, Twc=Twc, preds=preds, preds_depth=preds_depth, R=R, t=t)


def load_rendered_res_preds(file_name):
    with open(file_name, 'rb') as f:
        data = np.load(f)
        img = data['img']
        depth = data['depth']
        K = data['K']
        Twc = data['Twc']
        preds = data['preds']
        preds_depth = data['preds_depth']
        R = data['R']
        t = data['t']

    return img, depth, K, Twc, preds, preds_depth, R, t


def get_pt_from_img(img, depth, K, Twc):
    H, W = depth.shape
    x_2d = x_2d_coords(H, W).reshape((H * W, 2))
    X_3d = pi_inv(K, x_2d, depth.reshape((H * W, 1)))
    X_3d = transpose(Twc[:3, :3], Twc[:3, 3], X_3d)
    colors = img.reshape((H * W, 3)) / 255.0
    return X_3d, colors


def save_deform(file_name, img, depth, K, Twc, deform):
    with open(file_name, 'wb') as f:
        np.savez(f, img=img, depth=depth, K=K, Twc=Twc, deform=deform)


def load_deform(file_name):
    with open(file_name, 'rb') as f:
        data = np.load(f)
        img = data['img']
        depth = data['depth']
        K = data['K']
        Twc = data['Twc']
        deform = data['deform']

    return img, depth, K, Twc, deform


if __name__ == '__main__':
    # img, depth, K, Twc, preds, preds_depth, R, t = load_rendered_res_preds('rendered0.npz')
    # img1, depth1, K1, Twc1, preds1, preds_depth1, R1, t1 = load_rendered_res_preds('rendered1.npz')
    # H, W = depth.shape
    # vis = Visualizer(1280, 720)
    #
    # X_3d, colors = get_pt_from_img(img, depth, K, Twc)
    # vis.set_point_cloud(X_3d, colors, pt_size=3)
    #
    # lm_colors = np.ones((48-27, 3))
    # preds_X_3d = get_eyes_nose_3D_landmarks(img, depth, K, Twc, preds)
    # vis.set_point_cloud(preds_X_3d, lm_colors, pt_size=10, clear=False)
    #
    # # lm_colors[:, 0] = 0
    # # preds_X_3d = get_eyes_nose_3D_landmarks(img1, depth1, K1, Twc1, preds1)
    # # vis.set_point_cloud(preds_X_3d, lm_colors, pt_size=10, clear=False)
    #
    # T = np.eye(4, dtype=np.float32)
    # T[:3, :3] = R1
    # T[:3, 3] = t1
    # Twc1 = np.dot(T, Twc1)
    #
    # X_3d, colors = get_pt_from_img(img1, depth1, K1, Twc1)
    # vis.set_point_cloud(X_3d, colors, pt_size=3, clear=False)
    #
    # lm_colors[:, 1] = 0
    # preds_X_3d = get_eyes_nose_3D_landmarks(img1, depth1, K1, Twc1, preds1)
    # vis.set_point_cloud(preds_X_3d, lm_colors, pt_size=10, clear=False)
    #
    # vis.show()

    img, depth, K, Twc, deform = load_deform('deform.npz')
    H, W = depth.shape
    vis = Visualizer(1280, 720)

    X_3d, colors = get_pt_from_img(img, depth, K, Twc)
    vis.set_point_cloud(X_3d, colors, pt_size=3)
    X_3d_ = X_3d + deform.reshape((-1, 3))
    vis.set_point_cloud(X_3d_, np.ones((X_3d_.shape[0], 3)), pt_size=3, clear=False)

    vis.show()
