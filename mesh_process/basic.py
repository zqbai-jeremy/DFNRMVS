import numpy as np
import trimesh


def load_mesh_from_obj(obj_file_path, process=True):
    """
    Load the .obj file as a trimesh.base.Trimesh object
    :param obj_file_path:
    :param process: whether to pre-process the mesh when loading
    :return: a trimesh.base.Trimesh object
    """
    mesh = trimesh.load(obj_file_path, file_type='obj', process=process)
    mesh.apply_scale(1e-3)

    # Resize texture to power of 2
    H = mesh.visual.material.image.height
    W = mesh.visual.material.image.width

    def _nearest_pow2(v):
        # From http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
        # Credit: Sean Anderson
        v -= 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1

    # print(H, W)
    H = _nearest_pow2(H)
    W = _nearest_pow2(W)
    # print(H, W)
    mesh.visual.material.image = mesh.visual.material.image.resize((W, H))
    return mesh


def get_3D_rigid_transform(A, B):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A.T) + centroid_B.T

    return R, t
