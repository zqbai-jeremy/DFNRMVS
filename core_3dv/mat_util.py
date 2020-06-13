import torch


def batched_mat_inv(mat_tensor):
    """
    [TESTED, file: valid_banet_mat_inv.py]
    Returns the inverses of a batch of square matrices.
    alternative implementation: torch.gesv() with LU decomposition
    :param mat_tensor: Batched Square Matrix tensor with dim (N, m, m). N is the batch size, m is the size of square mat
    :return batched inverse square mat
    """
    n = mat_tensor.size(-1)
    flat_bmat_inv = torch.stack([m.inverse() for m in mat_tensor.view(-1, n, n)])
    return flat_bmat_inv.view(mat_tensor.shape)


def batched_mat_diag(diag):
    """
    [TESTED]
    Batched diagonal matrix generator
    :param diag: tensor stores the diagonal elements of RxR Matrix, dim (N, R), N is the batch size, R is the row of Mat
    :return: batched diagonal matrix
    """
    N = diag.shape[0]  # batch size
    r = diag.size(-1)  # square matrix row or column
    diag_mat = torch.stack([torch.diag(m) for m in diag.view(-1, r)])
    return diag_mat.view(N, r, r)


def batched_diag_vec(mat):
    """
    [TESTED]
    Batched diagonal matrix extractor
    :param diag: tensor stores the square mat of RxR Matrix, dim (N, R, R), N is the batch size, R is the row of Mat
    :return: batched diagonal vector of matrix
    """
    N = mat.shape[0]  # batch size
    r = mat.size(-1)  # square matrix row or column
    diag_mat = torch.stack([torch.diag(m) for m in mat.view(-1, r, r)])
    return diag_mat.view(N, r)
