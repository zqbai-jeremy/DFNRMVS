from numba import cuda, float32, int32
import ctypes
import torch
import numpy
import math




def cuda_grid_block_2d(H, W, default_threads_per_block=16):
    threads_per_block_2d = (default_threads_per_block, default_threads_per_block)
    block_per_grid_2d = (int(math.ceil(H / threads_per_block_2d[0])),
                         int(math.ceil(W / threads_per_block_2d[1])))
    return block_per_grid_2d, threads_per_block_2d


def get_device_ndarray(t, stream=None):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.cudadrv.driver.driver.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], numpy.dtype('float32'),
                                                  gpu_data=mp, stream=stream)


