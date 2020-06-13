import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender

from core_math.transfom import rotation_matrix
from visualizer.visualizer_2d import show_multiple_img
from mesh_process.basic import *


def K_from_PerspectiveCamera(camera, height, width):
    """
    Get intrinsic matrix K from pyrender PerspectiveCamera object
    :param camera: PerspectiveCamera object
    :param height: Image height
    :param width: Image width
    :return: K
    """
    mat = camera.get_projection_matrix(width, height)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = mat[0, 0] * (width // 2)
    K[0, 2] = width // 2
    K[1, 1] = mat[1, 1] * (height // 2)
    K[1, 2] = height // 2
    return K


def simple_face_rendering(obj_file_path, show=True):
    mesh = load_mesh_from_obj(obj_file_path)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 10.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],#/300],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # camera_pose = rotation_matrix(angle=np.pi / 4.0, direction=[0.0, 1.0, 0.0])
    # camera_pose[0, 3] = camera_pose[2, 3] = np.sqrt(2) / 2
    scene.add(camera, pose=camera_pose)

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    light_pose = rotation_matrix(angle=0.0, direction=[0.0, 1.0, 0.0])
    scene.add(light, pose=light_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(960, 1280)
    color, depth = r.render(scene)
    # depth[depth < 1e-5] = 0.75

    # Show the images
    if show:
        img_list = [{'img': color, 'title': 'RGB'},
                    {'img': depth, 'title': 'Depth'}]
        show_multiple_img(img_list, num_cols=2)

    # print(depth[480, 640])
    r.delete()

    # Compute camera pose Twc
    Twc = camera_pose
    T = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Twc = np.dot(T, np.dot(Twc, T))

    return color, depth, K_from_PerspectiveCamera(camera, 1280, 960), Twc


def face_rendering(mesh, camera_pose, light_poses, show=True):
    """
    Render face RGBD images with input camera pose and lighting
    :param mesh: Trimesh object
    :param camera_pose: Twc, np.array 4x4
    :param light_poses: list of light poses, Twc, list[np.array 4x4]
    :param show: whether show rendered image
    :return:
    """
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 10.0)
    scene.add(camera, pose=camera_pose)

    # Set up the light
    for light_pose in light_poses:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
        light_pose = rotation_matrix(angle=0.0, direction=[0.0, 1.0, 0.0])
        scene.add(light, pose=light_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(960, 1280)
    color, depth = r.render(scene)
    # depth[depth < 1e-5] = 0.75

    # Show the images
    if show:
        img_list = [{'img': color, 'title': 'RGB'},
                    {'img': depth, 'title': 'Depth'}]
        show_multiple_img(img_list, num_cols=2)

    # print(depth[480, 640])
    r.delete()

    # Compute camera pose Twc
    Twc = camera_pose
    T = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Twc = np.dot(T, np.dot(Twc, T))

    return color, depth, K_from_PerspectiveCamera(camera, 1280, 960), Twc

