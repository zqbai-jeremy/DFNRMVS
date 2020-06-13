import torch
import cv2
import numpy as np


def colormap(tensor: torch.tensor, cmap='jet', clip_range=None, scale_each=True, chw_order=True):
    """
    Create colormap for each single channel input map
    :param tensor: input single-channel image, dim (N, H, W) or (N, 1, H, W)
    :param cmap: the type of color map
    :param chw_order: the output type of tensor, either CHW or HWC
    :param clip_range: the minimal or maximal clip on input tensor
    :param scale_each: normalize the input based on each image instead of the whole batch
    :return: colormap tensor, dim (N, 3, H, W) if 'chw_order' is True or (N, H, W, 3)
    """
    if cmap == 'gray':
        cmap_tag = cv2.COLORMAP_BONE
    elif cmap == 'hsv':
        cmap_tag = cv2.COLORMAP_HSV
    elif cmap == 'hot':
        cmap_tag = cv2.COLORMAP_HOT
    elif cmap == 'cool':
        cmap_tag = cv2.COLORMAP_COOL
    else:
        cmap_tag = cv2.COLORMAP_JET

    if tensor.dim() == 2: # single image
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    elif tensor.dim() == 4:
        if tensor.size(1) == 1:
            tensor = tensor.view(tensor.size(0), tensor.size(2), tensor.size(3))
        else:
            raise Exception("The input image should has one channel.")
    elif tensor.dim() > 4:
        raise Exception("The input image should has dim of (N, H, W) or (N, 1, H, W).")

    # normalize
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if clip_range is not None:
        assert isinstance(clip_range, tuple), \
            "range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, clip_range)
    else:
        norm_range(tensor, clip_range)

    # apply color map
    N, H, W = tensor.shape
    color_tensors = []
    for n in range(N):
        sample = tensor[n, ...].detach().cpu().numpy()
        colormap_sample = cv2.applyColorMap((sample * 255).astype(np.uint8), cmap_tag)
        colormap_sample = cv2.cvtColor(colormap_sample, cv2.COLOR_BGR2RGB)
        color_tensors.append(torch.from_numpy(colormap_sample).cpu())
    color_tensors = torch.stack(color_tensors, dim=0).float() / 255.0

    return color_tensors.permute(0, 3, 1, 2) if chw_order else color_tensors


def heatmap_blend(img:torch.tensor, heatmap:torch.tensor, heatmap_blend_alpha=0.5, heatmap_clip_range=None, cmap='jet'):
    """
    Blend the colormap onto original image
    :param img: original image in RGB, dim (N, 3, H, W)
    :param heatmap: input heatmap, dim (N, H, W) or (N, 1, H, W)
    :param heatmap_blend_alpha: blend factor, 'heatmap_blend_alpha = 0' means the output is identical to original image
    :param cmap: colormap to blend
    :return: blended heatmap image, dim (N, 3, H, W)
    """
    if heatmap.dim() == 4:
        if heatmap.size(1) == 1:
            heatmap = heatmap.view(heatmap.size(0), heatmap.size(2), heatmap.size(3))
        else:
            raise Exception("The heatmap should be (N, 1, H, W) or (N, H, W)")
    N, C3, H, W = img.shape

    assert heatmap_blend_alpha < 1.0
    assert H == heatmap.size(1)
    assert W == heatmap.size(2)
    assert N == heatmap.size(0)
    assert C3 == 3                      # input image has three channel RGB

    color_map = colormap(heatmap, cmap=cmap, clip_range=heatmap_clip_range, chw_order=True).to(img.device)
    output_heat_map = img.clone()*(1.0 - heatmap_blend_alpha) + color_map * heatmap_blend_alpha
    return output_heat_map


