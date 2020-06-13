import numpy as np
import cv2
import torch
import torchvision
from PIL import Image

# External libs
import external.deep_head_pose.code.hopenet as hopenet
import external.deep_head_pose.code.utils as hopenet_utils

# Internal libs
from core_3dv.camera_operator import x_2d_coords, pi_inv, transpose
from mesh_process.basic import *


def get_eyes_nose_3D_landmarks(img, depth, K, Twc, preds):
    """
    Get the positions of 3D landmarks of eyes (lower bound) and nose
    :param img:
    :param depth:
    :param K: intrinsic
    :param Twc: camera pose
    :param preds: predicted 2D landmarks
    :return: pos: (num_of_landmarks, 3)
    """
    def get_pt_from_img(img, depth, K):
        H, W = depth.shape
        x_2d = x_2d_coords(H, W).reshape((H * W, 2))
        X_3d = pi_inv(K, x_2d, depth.reshape((H * W, 1)))
        colors = img.reshape((H * W, 3)) / 255.0
        return X_3d, colors

    H, W = depth.shape
    X_3d, colors = get_pt_from_img(img, depth, K)
    X_3d = transpose(Twc[:3, :3], Twc[:3, 3], X_3d)
    preds_X_3d = cv2.remap(X_3d.reshape((H, W, 3)), preds[27:48, 0], preds[27:48, 1], interpolation=cv2.INTER_LINEAR).\
                 reshape((preds[27:48, 0].shape[0], 3))
    preds_X_3d[9:11, :] = preds_X_3d[13:15, :]
    preds_X_3d[11:13, :] = preds_X_3d[19:, :]
    return preds_X_3d[:13, :]


def align_faces_3D(faces):
    """
    Align all faces to the first one with 3D landmarks of eyes and nose
    :param faces: list[tuple(image, depth, K, Twc, preds)]
    :return: R, t: lists of np.array
    """
    ref_img, ref_depth, ref_K, ref_Twc, ref_preds = faces[0]
    ref_lm = get_eyes_nose_3D_landmarks(ref_img, ref_depth, ref_K, ref_Twc, ref_preds)
    Rs = []
    ts = []
    for i, face in enumerate(faces):
        if i == 0:
            Rs.append(np.eye(3, dtype=np.float32))
            ts.append(np.zeros(3, dtype=np.float32))
            continue
        img, depth, K, Twc, preds = face
        lm = get_eyes_nose_3D_landmarks(img, depth, K, Twc, preds)
        R, t = get_3D_rigid_transform(lm, ref_lm)
        Rs.append(R)
        ts.append(t)
    return Rs, ts


def get_square_face_image(face_detector, img, scale, size):
    """
    Crop a square face region based on face_detector (SFD in https://github.com/1adrianb/face-alignment)
    :param face_detector: SFD face detector
    :param img: raw RGB image read by opencv
    :param scale: side length of the square = scale * max(detected bounding box width, detected bounding box height)
    :param size: resize to this size, (height, width)
    :return: square face image, pixel value in [0, 1], np.array(H, W, 3)
    """
    d = face_detector.detect_from_image(img[..., ::-1].copy())
    if len(d) == 0:
        return None
    d = d[0]
    center = [d[3] - (d[3] - d[1]) / 2.0, d[2] - (d[2] - d[0]) / 2.0]
    center[0] += (d[3] - d[1]) * 0.06
    l = max(d[2] - d[0], d[3] - d[1]) * scale
    x_s = int(center[1] - (l / 2) + 0.5)
    y_s = int(center[0] - (l / 2) + 0.5)
    x_e = int(center[1] + (l / 2) + 0.5)
    y_e = int(center[0] + (l / 2) + 0.5)
    img = Image.fromarray(img).crop((x_s, y_s, x_e, y_e))
    img = cv2.resize(np.asarray(img), (size[1], size[0])).astype(np.float32) / 255.0
    return img


def init_DeepHeadPose(model_path):
    """
    Initialize Deep Head Pose model https://github.com/natanielruiz/deep-head-pose
    :param model_path: path of stored model
    :return: Deep Head Pose model
    """
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict)
    model.cuda().eval()
    return model


def get_head_pose_by_DeepHeadPose(model, img):
    """
    Compute Euler angle of the head by Deep Head Pose https://github.com/natanielruiz/deep-head-pose
    :param model: Deep Head Pose model
    :param img: detected RGB face image, pixel value in [0, 1], np.array(H, W, 3)
    :return: pitch, yaw, roll in degree
    """
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img.transpose((2, 0, 1)))).unsqueeze(0).cuda()

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.tensor(idx_tensor).float().cuda()

    yaw, pitch, roll = model(img)
    yaw_predicted = hopenet_utils.softmax_temperature(yaw, 1)
    pitch_predicted = hopenet_utils.softmax_temperature(pitch, 1)
    roll_predicted = hopenet_utils.softmax_temperature(roll, 1)
    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = -pitch_predicted.item()
    yaw_predicted = -yaw_predicted.item()
    roll_predicted = -roll_predicted.item()

    return pitch_predicted, yaw_predicted, roll_predicted


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform
