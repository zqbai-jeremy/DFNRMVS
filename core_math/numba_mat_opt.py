from numba import cuda, float32
import numpy as np
import math
from core_math.numba_util import cuda_grid_block_2d


@cuda.jit(argtypes=[float32[:, :], float32[:, :], float32[:, :]], device=True, inline=True)
def mat3_mul(m0, m1, out):
    out[0, 0] = m0[0, 0] * m1[0, 0] + m0[0, 1] * m1[1, 0] + m0[0, 2] * m1[2, 0]
    out[0, 1] = m0[0, 0] * m1[0, 1] + m0[0, 1] * m1[1, 1] + m0[0, 2] * m1[2, 1]
    out[0, 2] = m0[0, 0] * m1[0, 2] + m0[0, 1] * m1[1, 2] + m0[0, 2] * m1[2, 2]
    out[1, 0] = m0[1, 0] * m1[0, 0] + m0[1, 1] * m1[1, 0] + m0[1, 2] * m1[2, 0]
    out[1, 1] = m0[1, 0] * m1[0, 1] + m0[1, 1] * m1[1, 1] + m0[1, 2] * m1[2, 1]
    out[1, 2] = m0[1, 0] * m1[0, 2] + m0[1, 1] * m1[1, 2] + m0[1, 2] * m1[2, 2]
    out[2, 0] = m0[2, 0] * m1[0, 0] + m0[2, 1] * m1[1, 0] + m0[2, 2] * m1[2, 0]
    out[2, 1] = m0[2, 0] * m1[0, 1] + m0[2, 1] * m1[1, 1] + m0[2, 2] * m1[2, 1]
    out[2, 2] = m0[2, 0] * m1[0, 2] + m0[2, 1] * m1[1, 2] + m0[2, 2] * m1[2, 2]


@cuda.jit(argtypes=[float32[:, :], float32[:], float32[:]], device=True, inline=True)
def mat3_vec3_mul(m, v, out):
    out[0] = m[0, 0] * v[0] + m[0, 1] * v[1] + m[0, 2] * v[2]
    out[1] = m[1, 0] * v[0] + m[1, 1] * v[1] + m[1, 2] * v[2]
    out[2] = m[2, 0] * v[0] + m[2, 1] * v[1] + m[2, 2] * v[2]


@cuda.jit(argtypes=[float32[:, :], float32[:], float32[:]], device=True, inline=True)
def mat34_homo_vec3_mul(m, v, out):
    out[0] = m[0, 0] * v[0] + m[0, 1] * v[1] + m[0, 2] * v[2] + m[0, 3]
    out[1] = m[1, 0] * v[0] + m[1, 1] * v[1] + m[1, 2] * v[2] + m[1, 3]
    out[2] = m[2, 0] * v[0] + m[2, 1] * v[1] + m[2, 2] * v[2] + m[2, 3]


@cuda.jit(argtypes=[float32[:, :], float32[:], float32[:]], device=True, inline=True)
def mat34_vec4_mul(m, v, out):
    out[0] = m[0, 0] * v[0] + m[0, 1] * v[1] + m[0, 2] * v[2] + m[0, 3] * v[3]
    out[1] = m[1, 0] * v[0] + m[1, 1] * v[1] + m[1, 2] * v[2] + m[1, 3] * v[4]
    out[2] = m[2, 0] * v[0] + m[2, 1] * v[1] + m[2, 2] * v[2] + m[2, 3] * v[5]


""" Test Routines ------------------------------------------------------------------------------------------------------
"""
@cuda.jit(argtypes=[float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :]])
def mat3_mul_kernel(m0, m1, out):
    i, j = cuda.grid(2)
    m0_mat = m0[i, j, :, :]
    m1_mat = m1[i, j, :, :]
    out_mat = out[i, j, :, :]
    mat3_mul(m0_mat, m1_mat, out_mat)
    # cuda.syncthreads()


@cuda.jit(argtypes=[float32[:, :, :, :], float32[:, :, :], float32[:, :, :]])
def mat3_mul_vec_kernel(m, v, out):
    i, j = cuda.grid(2)
    m_mat = m[i, j, :, :]
    v_vec = v[i, j, :]
    out_vec = out[i, j, :]
    mat3_vec3_mul(m_mat, v_vec, out_vec)


# TEST
def test_grid_multiply(A, B):
    H, W = A.shape[:2]
    blocks_2d, threads_2d = cuda_grid_block_2d(H, W)
    out = np.zeros_like(A, dtype=np.float32)
    stream = cuda.stream()
    with stream.auto_synchronize():
        d_A = cuda.to_device(A, stream)
        d_B = cuda.to_device(B, stream)
        d_out = cuda.to_device(out, stream)
        mat3_mul_kernel[blocks_2d, threads_2d, stream](d_A, d_B, d_out)
        d_out.to_host(stream)
    return out


def test_grid_mat_mul_vec(A, B):
    H, W = A.shape[:2]
    blocks_2d, threads_2d = cuda_grid_block_2d(H, W)
    out = np.zeros((H, W, 3), dtype=np.float32)
    stream = cuda.stream()
    with stream.auto_synchronize():
        d_A = cuda.to_device(A, stream)
        d_B = cuda.to_device(B, stream)
        d_out = cuda.to_device(out, stream)
        mat3_mul_vec_kernel[blocks_2d, threads_2d, stream](d_A, d_B, d_out)
        d_out.to_host(stream)
    return out


if __name__ == '__main__':
    # Test the matrix operators
    A = np.random.rand(240, 320, 3, 3).astype(np.float32)
    B = np.random.rand(240, 320, 3, 3).astype(np.float32)
    C = np.random.rand(240, 320, 3).astype(np.float32)
    # A = A.reshape((16, 16, 3*3))
    # B = B.reshape((16, 16, 3*3))
    out = test_grid_multiply(A, B)
    out_vec = test_grid_mat_mul_vec(A, C)

    # A = A.reshape((16, 16, 3, 3))
    # B = B.reshape((16, 16, 3, 3))
    out_array = np.zeros_like(A, dtype=np.float32)
    out_vec_array = np.zeros((240, 320, 3), dtype=np.float32)
    for i in range(0, 240):
        for j in range(0, 320):
            out_array[i, j, :, :] = np.dot(A[i, j, :, :], B[i, j, :, :])
            out_vec_array[i, j, :] = np.dot(A[i, j, :, :], C[i, j, :])

    print(out[3, 4])
    print(out_array[3, 4])
    print(out_vec[4, 4])
    print(out_vec_array[4, 4])