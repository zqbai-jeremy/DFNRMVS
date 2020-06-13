import numpy as np
import torch
from core_3dv.mat_util import *

""" Gauss Newton Optimizer
"""
def gauss_newtown_update(J, r):
    """
    Gauss-Newtown Update
    :param J: Jacobin Matrix J(x) of residual error function r(x), dim: (N, n_r_out, n_r_in)
    :param r: residual error function, dim: (N, n_r_out)
    :return delta_x: update vector, dim: (N, n_r_in)
    :return delta_x_norm: norm of the update vector, dim: (N, 1)
    """
    N = J.shape[0]  # batch size
    n_f_in = J.shape[2]
    n_f_out = J.shape[1]

    # Compute Update Vector: - (J^tJ)^{-1} J^tR
    Jt = J.transpose(1, 2)  # batch transpose (H,W) to (W, H), dim: (N, n_f_in, n_f_out)
    JtJ = torch.bmm(Jt, J)  # dim: (N, n_f_in, n_f_in)
    JtR = torch.bmm(Jt, r.view(N, n_f_out, 1))  # dim: (N, n_f_in, 1)

    delta_x = - torch.bmm(batched_mat_inv(JtJ), JtR).view(N, n_f_in)  # dim: (N, n_f_in)
    delta_x_norm = torch.sqrt(torch.sum(delta_x * delta_x, dim=1)).detach()  # dim: (N, 1)

    return delta_x, delta_x_norm


def levenberg_marquardt_update(J, r, lambda_weight):
    """
    Levenberg Marquardt Update
    :param J: Jacobin Matrix J(x) of residual error function r(x), dim: (N, n_r_out, n_r_in)
    :param r: residual error function, dim: (N, n_r_out)
    :param labmda_weight: the damping vector, dim: (N, n_r_in)
    :return delta_x: update vector, dim: (N, n_r_in)
    :return delta_x_norm: norm of the update vector, dim: (N, 1)
    """
    N = J.shape[0]  # batch size
    n_f_in = J.shape[2]
    n_f_out = J.shape[1]

    # Compute Update Vector: - (J^{T}J + \lambda diag(J^{T}J) )^{-1} J^{T}R
    Jt = J.transpose(1, 2)  # batch transpose (H,W) to (W, H), dim: (N, n_f_in, n_f_out)
    JtJ = torch.bmm(Jt, J)  # dim: (N, n_f_in, n_f_in)
    JtR = torch.bmm(Jt, r.view(N, n_f_out, 1))  # dim: (N, n_f_in, 1)

    JtJ_lambda_diag = JtJ + \
                      batched_mat_diag(lambda_weight * (batched_diag_vec(JtJ)))  # dim: (N, n_f_in)

    delta_x = - torch.bmm(batched_mat_inv(JtJ_lambda_diag), JtR).view(N, n_f_in)  # dim: (N, n_f_in)
    delta_x_norm = torch.sqrt(torch.sum(delta_x * delta_x, dim=1)).detach()  # dim: (N, 1)

    return delta_x, delta_x_norm


def gauss_newton(f, Jac, x0, eps=1e-4, max_itr=20, verbose=False):
    """
    Reference: https://blog.xiarui.net/2015/01/22/gauss-newton/
    :param f: residual error computation, output out dim: (N, n_f_out)
    :param Jac: jacobi matrix of input parameter, out dim: (N, n_f_out, n_f_in)
    :param x0: initial guess of parameter, dim: (N, n_f_in)
    :param eps: stop condition, when eps > norm(delta), where delta is the update vector
    :param max_itr: maximum iteration
    :param verbose: print the iteration information
    :return: x: optimized parameter
    :return: boolean: optimization converged
    """

    N = x0.shape[0]  # batch size
    n_f_in = x0.shape[1]  # input parameters

    r = f(x0)  # residual error r(x0), dim: (N, n_f_out)
    n_f_out = r.shape[1]

    lambda_w = 0.05 * torch.ones((N, n_f_in))

    # Iterative optimizer
    x = x0
    converged_flag = True
    for itr in range(0, max_itr):

        # Compute the Jacobi with respect to the residual error
        J = Jac(x)

        # Update the
        delta_x, delta_norm = gauss_newtown_update(J, r)
        # delta_x, delta_norm = levenberg_marquardt_update(J, r, lambda_w)

        # Update parameter
        x = x + delta_x

        sum_delta_norm = torch.max(delta_norm).item()
        if sum_delta_norm < eps:
            converged_flag = True
            break
        r = f(x)

        if verbose:
            print('[Gauss-Newton Optimizer ] itr=%d, r_norm: %f, update_norm: %f' % (
            itr, torch.bmm(r.view(N, 1, r.shape[1]), r.view(N, r.shape[1], 1))[0].numpy(), sum_delta_norm))

    return x, converged_flag


''' Test ---------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':
    def f(input_x):
        input = input_x.cpu().numpy()
        x = input[0]
        y = input[1]
        z = input[2]
        e = np.asarray([[x[0] ** 2 + x[0] * x[1] - 10, x[1] + 3 * x[0] * x[1] * x[1] - 57],  # [2, 3]
                        [y[0] ** 2 + y[0] * y[1] - 10, y[1] ** 2 + 3 * y[0] * y[1] + 21],
                        [z[0] * z[1] - 12, z[0] ** 2 + z[1] - 19]], dtype=np.float32)  # [4.27095629 -1.92956028]

        return torch.from_numpy(e).view((3, 2))


    def Jac(input_x):
        input = input_x.cpu().numpy()
        x = input[0]
        y = input[1]
        z = input[2]
        J_m = np.asarray([[[2 * x[0] + x[1], x[0]], [3 * x[1] * x[1], 1 + 6 * x[0] * x[1]]],
                          [[2 * y[0] + y[1], y[0]], [3 * y[1], 2 * y[1] + 3 * y[0]]],
                          [[z[1], z[0]], [2 * z[0], 1.0]]], dtype=np.float32)
        return torch.from_numpy(J_m).view((3, 2, 2))


    torch.set_default_tensor_type('torch.FloatTensor')

    x0 = np.asarray([[1.0, 1.5], [-1.0, -2.0], [1.0, 1.0]], dtype=np.float32).reshape((3, 2))
    x0 = torch.from_numpy(x0)
    x_opt = gauss_newton(f, Jac, x0, verbose=True)
    print(x_opt)