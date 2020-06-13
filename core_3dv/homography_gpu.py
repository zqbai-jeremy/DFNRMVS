import torch


def d2z(K, n, d, x2d):
    """
    Compute plane parameter of z, ref. [Luwei Master Thesis, page:18]
    :param K: camera intrinsic matrix, dim: (N, 3, 3)
    :param n: surface normal, dim: (N, M, 3), M is the number of samples
    :param d: depth value, dim: (N, M, 1) or (N, M)
    :param x2d: image plane coordinates, dim: (N, M, 3) or (N, M, 2)
    :param z: plane parameter, dim: (N, M, 1)
    """
    assert n.shape[1] == d.shape[1] and n.shape[1] == x2d.shape[1]
    assert n.dim() == 3
    assert d.dim() == 2 or d.dim() == 3

    N, M = n.shape[0], n.shape[1]
    d = d.view((N, M, 1))
    x2d = x2d[:, :, :2].view((N, M, 2))

    f, cx, cy = K[:, 0:1, 0:1], K[:, 0:1, 2:3], K[:, 1:2, 2:3]                                      # dim: (N, 1, 1)

    # compute s = [px-cx, py-cy, f]
    s = torch.cat([
        x2d[:, :, 0:1] - cx,                                                                        # dim: (N, M, 1)
        x2d[:, :, 1:2] - cy,                                                                        # dim: (N, M, 1)
        f.repeat(1, M, 1)                                                                           # dim: (N, M, 1)
    ], dim=-1)                                                                                      # dim: (N, M, 3)

    # compute dot product of l=sTn
    l = torch.einsum('nmjk,nmkl->nmjl',
                     s.view(N, M, 1, 3),
                     n.view(N, M, 3, 1)).view((N, M, 1))                                            # dim: (N, M, 1)

    # compute z = -l*d/f
    z = -torch.div(torch.mul(l, d), f)                                                              # dim: (N, M, 1)

    return z


def homography_a2b(K_b, K_a_inv, R, t, n, z):
    """
    Compute homography from a to b that induced by
    plane parameter (nT, z). ref. [Luwei Master Thesis, page:18]
    see 'test_homography.ipynb' for example.
    :param K_b: camera intrinsic matrix of camera b, dim: (N, 3, 3)
    :param K_a_inv: inverse camera intrinsic matrix of camera a, dim: (N, 3, 3)
    :param R: relative rotation matrix from A to B
    :param t: relative translation vector from A to B
    :param n: surface normal, dim: (N, M, 3)
    :param z: plane parameter z, use function d2z() to compute, dim: (N, M, 1)
    :return H: Homography matrix, dim: (N, M, 3, 3)
    """
    assert n.dim() == 3
    assert z.shape[1] == n.shape[1]
    N, M = n.shape[0], n.shape[1]

    c = R.view((N, 1, 3, 3)) - torch.einsum('ijk,itkl->itjl',
                                            t.view((N, 3, 1)),
                                            n.view((N, M, 1, 3))) / z.view((N, M, 1, 1))            # dim: (N, M, 3, 3)
    H = torch.einsum('ijk,itkl->itjl', K_b, c)                                                      # dim: (N, M, 3, 3)
    H = torch.einsum('itjk,ikl->itjl', H, K_a_inv)                                                  # dim: (N, M, 3, 3)
    return H
