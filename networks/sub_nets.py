import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import scipy.io as sio
import os
import itertools
import shutil

# External libs
from external.face3d.face3d.morphable_model import MorphabelModel
from external.face3d.face3d.morphable_model.load import load_BFM_info

# Internal libs
from core_dl.base_net import BaseNet
import core_3dv.camera_operator_gpu as cam_opt
from networks.basic_feat_extrator import RGBNet
import data.BFM.utils as bfm_utils


def batched_gradient(features):
    """
    Compute gradient of a batch of feature maps
    :param features: a 3D tensor for a batch of feature maps, dim: (N, C, H, W)
    :return: gradient maps of input features, dim: (N, ２*C, H, W), the last row and column are padded with zeros
             (N, 0:C, H, W) = dI/dx, (N, C:2C, H, W) = dI/dy
    """
    H = features.size(-2)
    W = features.size(-1)
    C = features.size(1)
    N = features.size(0)
    grad_x = (features[:, :, :, 2:] - features[:, :, :, :W - 2]) / 2.0
    grad_x = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')
    grad_y = (features[:, :, 2:, :] - features[:, :, :H - 2, :]) / 2.0
    grad_y = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')
    grad = torch.cat([grad_x.view(N, C, H, W), grad_y.view(N, C, H, W)], dim=1)
    return grad


BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, n_gn=8):
        super(BasicBlockGN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.GroupNorm(n_gn, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.GroupNorm(n_gn, planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class RegressionNet(BaseNet):
    """
    Regress the shared params: shape. and the per view params: expression & pose
    """

    def __init__(self, n_shape_para=199, n_exp_para=29):
        """
        :param n_shape_para: number of params for shape
        :param n_exp_para: number of params for expression
        :return:
        """
        super(RegressionNet, self).__init__()
        self.n_shape_para = n_shape_para
        self.n_exp_para = n_exp_para

        self.rgb_net = RGBNet(input_dim=(3, 256, 256))

        skip = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(16, 256),
        )
        self.shared_branch = nn.Sequential(
            BasicBlockGN(512, 256, downsample=skip, n_gn=16)
        )
        self.iden_fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256, bias=False),    # FC: 256x8x8 --> 256
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_shape_para, bias=True)
        )

        skip = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(16, 256),
        )
        self.per_view_branch = nn.Sequential(
            BasicBlockGN(512 + 256, 256, downsample=skip, n_gn=16)
        )
        self.exp_pose_fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256, bias=False),    # FC: 256x8x8 --> 256
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.exp_fc = nn.Linear(256, n_exp_para, bias=True)
        self.angles_fc = nn.Linear(256, 3, bias=True)
        self.t_fc = nn.Linear(256, 2, bias=True)
        self.s_fc = nn.Linear(256, 1, bias=True)

        self.iden_fc[3].weight.data.fill_(0.0)
        self.exp_fc.weight.data.fill_(0.0)
        self.angles_fc.weight.data.fill_(0.0)
        self.t_fc.weight.data.fill_(0.0)
        self.s_fc.weight.data.fill_(0.0)
        self.iden_fc[3].bias.data.fill_(0.0)
        self.exp_fc.bias.data.fill_(0.0)
        self.angles_fc.bias.data.fill_(0.0)
        self.t_fc.bias.data.fill_(0.0)
        self.s_fc.bias.data.fill_(0.0)

    def train(self, mode=True):
        super(RegressionNet, self).train(mode)
        for module in self.rgb_net.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()

    def forward(self, x):
        """
        :param x: (N, V, C, H, W) images
        :return:
        """
        # Extract multi-level features
        N, V, C, H, W = x.shape
        rgb_feats = self.rgb_net(x.view(N * V, C, H, W))                            # dim: [(N * V, c, h, w)]

        # Shared branch forward
        rgb_feat8x8 = rgb_feats[1].view(N, V, 512, 8, 8)
        avg_feat8x8 = torch.mean(rgb_feat8x8, dim=1)                                # dim: (N, 512, 8, 8)
        sp_feat = self.shared_branch(avg_feat8x8)                                   # dim: (N, 256, 8, 8)
        sp_norm = self.iden_fc(sp_feat.view(N, 256 * 8 * 8))
        sp_norm[:, 80:] = 0.

        # Per view branch forward
        sp_feat = sp_feat.view(N, 1, 256, 8, 8).expand(N, V, 256, 8, 8).contiguous().view(N * V, 256, 8, 8)
        cat_feat = torch.cat([rgb_feats[1], sp_feat], dim=1)                        # dim: (N * V, 512 + 256, 8, 8)
        per_view_feat = self.per_view_branch(cat_feat)                              # dim: (N * V, 256, 8, 8)
        exp_pose_feat = self.exp_pose_fc(per_view_feat.view(N * V, 256 * 8 * 8))    # dim: (N * V, 256)
        exp_norm = self.exp_fc(exp_pose_feat)
        angles = self.angles_fc(exp_pose_feat)
        t = self.t_fc(exp_pose_feat)
        s = self.s_fc(exp_pose_feat)
        pose = torch.cat([angles, s, t], dim=1)

        return sp_norm.view(N, self.n_shape_para, 1), exp_norm.view(N, V, self.n_exp_para, 1), pose.view(N, V, 6),\
               rgb_feats


class BFMDecoder(BaseNet):
    """
    Decode BFM params to per vertex color
    """

    def __init__(self, MM_base_dir='./external/face3d/examples/Data/BFM'):
        super(BFMDecoder, self).__init__()

        # Initialize BFM
        # self.bfm = MorphabelModel(os.path.join(MM_base_dir, 'Out/BFM.mat'))
        # params_attr = sio.loadmat(os.path.join(MM_base_dir, '3ddfa/3DDFA_Release/Matlab/params_attr.mat'))
        # self.bfm.params_mean_3dffa = params_attr['params_mean']
        # self.bfm.params_std_3dffa = params_attr['params_std']
        # sigma_exp = sio.loadmat(os.path.join(MM_base_dir, 'sigma_exp.mat'))
        # self.bfm.sigma_exp = sigma_exp['sigma_exp'].reshape((29, 1))
        # self.bfm.face_region_mask = bfm_utils.get_tight_face_region(self.bfm, MM_base_dir, True)
        # self.bfm.tri_idx = bfm_utils.get_adjacent_triangle_idx(int(self.bfm.nver), self.bfm.model['tri'])
        self.bfm = bfm_utils.load_3DMM(MM_base_dir)
        model_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
        photo_weights = np.minimum(model_info['segbin'][1, :] + model_info['segbin'][2, :], 1)
        self.photo_weights = torch.from_numpy(photo_weights).view(1, 1, -1).float()

        self.bfm_torch = bfm_utils.MorphabelModel_torch(self.bfm)

    def cuda(self, device=None):
        super(BFMDecoder, self).cuda(device)
        self.photo_weights = self.photo_weights.cuda(device)
        self.bfm_torch.cuda(device)

    def forward(self, images, sp_norm, ep_norm, pose, ori_images):
        """
        Sample per vertex color
        :param images: (N, V, C, H, W)
        :param sp_norm: (N, n_shape_para, 1)
        :param ep_norm: (N, V, n_exp_para, 1)
        :param pose: (N, V, 6)
        :param ori_images: (N, V, C, H, W)
        :return: colors: per vertex color. (N, V, C, nver)
                 vis_mask: per vertex visibility. (N, V, 1, nver) byte
        """
        N, V, C, H, W = images.shape

        # Process params
        sp_norm = sp_norm.view(N, 1, self.bfm.n_shape_para, 1)\
                         .expand(N, V, self.bfm.n_shape_para, 1).contiguous()\
                         .view(N * V, self.bfm.n_shape_para, 1)
        ep_norm = ep_norm.view(N * V, self.bfm.n_exp_para, 1)
        pose = pose.view(N * V, 6)
        pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm = \
            pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3], pose[:, 4], pose[:, 5]
        pitch, yaw, roll, s, tx, ty = \
            bfm_utils.denormalize_pose_params(pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm)
        sp, ep, _ = self.bfm_torch.denormalize_BFM_params(sp_norm, ep_norm, None, 'normal_manual')

        # Process vertices
        vert = self.bfm_torch.generate_vertices(sp, ep)
        vert[:, :, 2] -= 7.5e4
        angles = torch.stack([pitch, yaw, roll], dim=1)
        zeros = torch.zeros_like(tx)
        t = torch.stack([tx, ty, zeros], dim=1)
        vert = self.bfm_torch.transform(vert, s, angles, t)

        # Sample colors
        images = images.view(N * V, C, H, W)
        ori_images = ori_images.view(N * V, C, H, W)
        vert_img = self.bfm_torch.to_image(vert, H, W)
        grid = cam_opt.x_2d_normalize(H, W, vert_img[:, :, :2].clone()).view(N * V, int(self.bfm.nver), 1, 2)
        colors = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros')
        ori_colors = F.grid_sample(ori_images, grid.detach(), mode='bilinear', padding_mode='zeros')

        # Get visibility map
        vis_mask = self.bfm_torch.backface_culling_cpu(vert, self.bfm.model['tri'])
        full_face_vis_mask = torch.from_numpy(np.copy(vis_mask)).to(images.device)          # dim: (N * V, nver, 1)
        vis_mask = np.logical_and(vis_mask, self.bfm.face_region_mask[np.newaxis, ...])
        vis_mask = torch.from_numpy(vis_mask).to(images.device)                             # dim: (N * V, nver, 1)

        return colors.view(N, V, C, int(self.bfm.nver)),\
               vis_mask.byte().view(N, V, 1, int(self.bfm.nver)),\
               full_face_vis_mask.byte().view(N, V, 1, int(self.bfm.nver)),\
               vert_img.view(N, V, int(self.bfm.nver), 3),\
               ori_colors.view(N, V, C, int(self.bfm.nver))


class MyGridSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grid, feat):
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True).detach()
        ctx.save_for_backward(feat, grid)
        return vert_feat

    @staticmethod
    def backward(ctx, grad_output):
        feat, grid = ctx.saved_tensors

        # Gradient for grid
        N, C, H, W = feat.shape
        _, Hg, Wg, _ = grid.shape
        feat_grad = batched_gradient(feat)      # dim: (N, 2*C, H, W)
        grid_grad = F.grid_sample(feat_grad, grid, mode='bilinear', padding_mode='zeros', align_corners=True)       # dim: (N, 2*C, Hg, Wg)
        grid_grad = grid_grad.view(N, 2, C, Hg, Wg).permute(0, 3, 4, 2, 1).contiguous()         # dim: (N, Hg, Wg, C, 2)
        grad_output_perm = grad_output.permute(0, 2, 3, 1).contiguous()                         # dim: (N, Hg, Wg, C)
        grid_grad = torch.bmm(grad_output_perm.view(N * Hg * Wg, 1, C),
                              grid_grad.view(N * Hg * Wg, C, 2)).view(N, Hg, Wg, 2)
        grid_grad[:, :, :, 0] = grid_grad[:, :, :, 0] * (W - 1) / 2
        grid_grad[:, :, :, 1] = grid_grad[:, :, :, 1] * (H - 1) / 2

        # Gradient for feat
        feat_d = feat.detach()
        feat_d.requires_grad = True
        grid_d = grid.detach()
        grid_d.requires_grad = True
        with torch.enable_grad():
            vert_feat = F.grid_sample(feat_d, grid_d, mode='bilinear', padding_mode='zeros', align_corners=True)
            vert_feat.backward(grad_output.detach())
        feat_grad = feat_d.grad

        return grid_grad, feat_grad


class FPNBlock(nn.Module):
    def __init__(self, in_nch, out_nch, n_gn):
        super(FPNBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv = nn.Conv2d(in_nch, out_nch, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv = BasicBlockGN(out_nch, out_nch, n_gn=n_gn)

    def forward(self, pre_feat, lateral_feat):
        pre_feat = self.up(pre_feat)
        lateral_feat = self.lateral_conv(lateral_feat)
        merge_feat = pre_feat + lateral_feat
        out_feat = self.out_conv(merge_feat)
        return out_feat, merge_feat


class FPN(nn.Module):
    def __init__(self, in_nchs=(128, 64, 32, 16), out_nch=64, n_gn=8):
        super(FPN, self).__init__()
        self.coarsest_conv = nn.Conv2d(in_nchs[0], out_nch, kernel_size=1, stride=1, padding=0, bias=False)
        self.n_layers = 0
        for i in range(1, len(in_nchs)):
            in_nch = in_nchs[i]
            setattr(self, 'layer%d' % (i - 1), FPNBlock(in_nch, out_nch, n_gn))
            self.n_layers += 1

    def forward(self, rgb_feats):
        fpn_feats = []
        pre_feat = self.coarsest_conv(rgb_feats[0])
        for i in range(self.n_layers):
            layer = getattr(self, 'layer%d' % i)
            fpn_feat, pre_feat = layer(pre_feat, rgb_feats[i + 1])
            fpn_feats.append(fpn_feat)
        return fpn_feats


class StepSizeNet(nn.Module):
    def __init__(self, in_nchs=(128, 2), nch=128, out_nchs=(199, 29, 6)):
        super(StepSizeNet, self).__init__()
        self.in_nchs = in_nchs
        self.nch = nch
        self.out_nchs = out_nchs
        in_nch = np.sum(np.asarray(in_nchs))
        out_nch = np.sum(np.asarray(out_nchs))
        self.net = nn.Sequential(
            nn.Linear(in_nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, out_nch),
            nn.Tanh()
        )
        self.net[-2].weight.data.fill_(0.0)
        self.net[-2].bias.data.fill_(0.0)

    def forward(self, inputs, scales, biases):
        """
        Predict step size of gradient descent
        :param inputs: (abs_residual (N, C), ...)
        :param scales: (scale_apply_to_step_size, ...)
        :param biases: (bias_apply_to_step_size, ...)
        :return: (step_size_of_params (N, n_params), ...)
        """
        _in = torch.cat(inputs, dim=1)
        _out = self.net(_in)
        start_i = 0
        outs = []
        for i in range(len(self.out_nchs)):
            out_nch = self.out_nchs[i]
            s = scales[i]
            b = biases[i]
            out = torch.pow(10., s * _out[:, start_i : start_i + out_nch] + b)
            outs.append(out)
            start_i += out_nch
        return outs, _out


class AdaptiveBasisNet(nn.Module):
    def __init__(self, n_adap_para, in_planes, n_planes, size, bfm, bfm_torch, n_gn, exp_basis_init):
        super(AdaptiveBasisNet, self).__init__()
        self.n_adap_para = n_adap_para
        self.size = size
        self.bfm = bfm
        self.bfm_torch = bfm_torch
        self.exp_basis_init = exp_basis_init
        self.pixel_vert_idx = getattr(bfm, 'pixel_vert_idx_' + str(size))
        self.pixel_vert_weights = getattr(bfm_torch, 'pixel_vert_weights_' + str(size))
        self.uv_coords = getattr(bfm_torch, 'uv_coords_' + str(size))

        skip = nn.Sequential(
            nn.Conv2d(in_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(n_gn, n_planes)
        )
        self.per_view_net = nn.Sequential(
            BasicBlockGN(in_planes, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip, n_gn=n_gn),
            BasicBlockGN(n_planes, n_planes, stride=1, dilation=(2, 2), residual=True, n_gn=n_gn),
            BasicBlockGN(n_planes, n_planes, stride=1, dilation=(4, 4), residual=True, n_gn=n_gn),
            BasicBlockGN(n_planes, n_planes, stride=1, dilation=(8, 8), residual=True, n_gn=n_gn)
        )
        self.net = nn.Sequential(
            nn.Conv2d(n_planes, n_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(n_gn, n_planes),
            nn.ReLU(True),
            nn.Conv2d(n_planes, n_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(n_gn, n_planes),
            nn.ReLU(True),
            nn.Conv2d(n_planes, 3 * n_adap_para, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.adap_bias_net = nn.Conv1d(n_adap_para, 1, 1, bias=False)

        if exp_basis_init:
            self.net[-1].weight.data.fill_(0.0)
            self.adap_bias_net.weight.data.fill_(0.0)
            self.apEV = nn.Parameter(bfm_torch.model['expEV'].clone())
            self.adapBias = nn.Parameter(bfm_torch.model['expPC'].clone())
        else:
            self.apEV = nn.Parameter(torch.ones((1, n_adap_para, 1)) * 1e3)

    def cuda(self, device=None):
        super(AdaptiveBasisNet, self).cuda(device)
        self.pixel_vert_weights = self.pixel_vert_weights.cuda(device)
        self.uv_coords = self.uv_coords.cuda(device)

    def denormalize_ap_norm(self, ap_norm):
        ap = ap_norm * self.apEV
        return ap

    def sample_per_vert_feat(self, feat, vert, H_img, W_img):
        """
        Project 3D vertices onto the feature map and sample feature vector for each 3D vertex
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: vert_feat: feature per vertex. (N, V, C, nver)
        """
        N, V, C, H, W = feat.shape
        feat = feat.view(N * V, C, H, W)
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Sample features
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img)
        grid = cam_opt.x_2d_normalize(H_img, W_img, vert_img[:, :, :2].clone()).view(N * V, int(self.bfm.nver), 1, 2)
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return vert_feat.view(N, V, C, int(self.bfm.nver))

    def forward(self, feat, vert, vis_mask, H_img, W_img):
        """
        Compute adaptive basis based on current reconstruction (i.e. vert)
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param vis_mask: visibility mask. (N, V, 1, nver)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: adap_B: adaptive basis. (N, nver * 3, n_adap_para)
        """
        N, V, C, _, _ = feat.shape
        nver = vert.shape[2]
        vert_feat = self.sample_per_vert_feat(feat, vert, H_img, W_img)             # (N, V, C, nver)
        nonface_region_mask = torch.from_numpy(np.invert(self.bfm.face_region_mask.copy())) \
                              .to(vert.device).view(1, 1, 1, nver)
        vert_feat_masked = vert_feat * vis_mask.float() + vert_feat * nonface_region_mask.float()
        vert_pos = vert.transpose(2, 3).contiguous() / (H_img + W_img) * 2.         # (N, V, 3, nver)
        vert_feat = torch.cat([vert_feat_masked, vert_pos], dim=2).view(N * V, C + 3, nver)

        # Render to UV space
        pixel_vert_feat = vert_feat[:, :, self.pixel_vert_idx]                      # (N * V, C + 3, size, size, 3)
        pixel_vert_weighted_feat = pixel_vert_feat * self.pixel_vert_weights.view(1, 1, self.size, self.size, 3)
        uv_per_view_feat = torch.sum(pixel_vert_weighted_feat, dim=-1)              # (N * V, C + 3, size, size)

        # Conv to adaptive basis
        uv_per_view_feat = self.per_view_net(uv_per_view_feat).view(N, V, C, self.size, self.size)
        (uv_feat, _) = torch.max(uv_per_view_feat, dim=1)                           # (N, C, size, size)
        adap_B_uv = self.net(uv_feat)                                               # (N, 3 * n_adap_para, size, size)
        grid = cam_opt.x_2d_normalize(self.size, self.size, self.uv_coords[:, :, :2].clone()) \
            .view(1, nver, 1, 2).expand(N, nver, 1, 2)
        adap_B = F.grid_sample(adap_B_uv, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (N, 3 * n_adap_para, nver, 1)
        adap_B = adap_B.permute(0, 2, 1, 3).contiguous().view(N, nver * 3, self.n_adap_para)
        if self.exp_basis_init:
            adap_B = adap_B + self.adapBias

        # Initialize adaptive displacement
        # adap_bias = self.adap_bias_net(adap_B.permute(0, 2, 1)).view(N, 1, nver, 3)
        adap_bias = None

        return adap_B, adap_bias, adap_B_uv


class NRMVSOptimization(BaseNet):

    def __init__(self, opt_step_size=1e-5, MM_base_dir='./external/face3d/examples/Data/BFM'):
        super(NRMVSOptimization, self).__init__()
        self.opt_step_size = opt_step_size
        self.training = True

        # Initialize BFM
        # self.bfm = MorphabelModel(os.path.join(MM_base_dir, 'Out/BFM.mat'))
        # params_attr = sio.loadmat(os.path.join(MM_base_dir, '3ddfa/3DDFA_Release/Matlab/params_attr.mat'))
        # self.bfm.params_mean_3dffa = params_attr['params_mean']
        # self.bfm.params_std_3dffa = params_attr['params_std']
        # sigma_exp = sio.loadmat(os.path.join(MM_base_dir, 'sigma_exp.mat'))
        # self.bfm.sigma_exp = sigma_exp['sigma_exp'].reshape((29, 1))
        # self.bfm.face_region_mask = bfm_utils.get_tight_face_region(self.bfm, MM_base_dir, True)
        # self.bfm.tri_idx = bfm_utils.get_adjacent_triangle_idx(int(self.bfm.nver), self.bfm.model['tri'])
        self.bfm = bfm_utils.load_3DMM(MM_base_dir)
        model_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
        photo_weights = np.minimum(model_info['segbin'][1, :] + model_info['segbin'][2, :], 1)
        self.photo_weights = torch.from_numpy(photo_weights).view(1, 1, -1).float()

        self.bfm_torch = bfm_utils.MorphabelModel_torch(self.bfm)

        # Feature extractor
        self.rgb_net = RGBNet(input_dim=(3, 256, 256))
        # self.fpn = FPN(in_nchs=(512, 512, 256, 128, 64), out_nch=128, n_gn=8)
        self.fpn = FPN(in_nchs=(512, 512, 256, 128, 64, 32), out_nch=128, n_gn=8)
        self.basis_fpn = FPN(in_nchs=(512, 512, 256, 128, 64, 32), out_nch=128, n_gn=8)

        # Step size network
        self.step_net2 = StepSizeNet(in_nchs=(128, 2), nch=128,
                                     # out_nchs=(self.bfm.n_shape_para, self.bfm.n_exp_para, 6))
                                     out_nchs=(1, 1, 1, 1))
        self.step_net3 = StepSizeNet(in_nchs=(128, 2), nch=128,
                                     # out_nchs=(1,))
                                     out_nchs=(1, 1, 1, 1))
        self.step_net4 = StepSizeNet(in_nchs=(128, 2), nch=128,
                                     # out_nchs=(1,))
                                     out_nchs=(1, 1, 1, 1))

        # Adaptive basis generator
        self.adap_B_net2 = AdaptiveBasisNet(64, 128 + 3, 128, 32, self.bfm, self.bfm_torch, 8, True)
        self.adap_B_net3 = AdaptiveBasisNet(64, 128 + 3, 128, 64, self.bfm, self.bfm_torch, 8, False)
        self.adap_B_net4 = AdaptiveBasisNet(64, 128 + 3, 128, 128, self.bfm, self.bfm_torch, 8, False)
        self.adap_B = None
        self.adap_bias = None
        self.n_adap_para = None
        self.adap_B_net = None
        self.pre_adap_delta = None

    def cuda(self, device=None):
        super(NRMVSOptimization, self).cuda(device)
        self.photo_weights = self.photo_weights.cuda(device)
        self.bfm_torch.cuda(device)
        self.adap_B_net2.cuda(device)
        self.adap_B_net3.cuda(device)
        self.adap_B_net4.cuda(device)

    def train(self, mode=True):
        super(NRMVSOptimization, self).train(mode)

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.rgb_net.apply(set_bn_eval)

    @torch.enable_grad()
    def compute_vert_from_params(self, sp_norm, ep_norm, pose, ap_norm):
        """
        Use BFM to compute positions of vertices of face mesh from params
        :param sp_norm: (N, n_shape_para, 1)
        :param ep_norm: (N, V, n_exp_para, 1)
        :param pose: (N, V, 6)
        :param ap_norm: (N, V, n_adap_para, 1)
        :return: vert: mesh vertices. (N, V, nver, 3)
        """
        N, V, _ = pose.shape

        # Process params
        sp_norm = sp_norm.view(N, 1, self.bfm.n_shape_para, 1) \
                         .expand(N, V, self.bfm.n_shape_para, 1).contiguous() \
                         .view(N * V, self.bfm.n_shape_para, 1)
        # sp_norm[:, 80:, :] = 0.
        ep_norm = ep_norm.view(N * V, self.bfm.n_exp_para, 1)
        ep_norm = ep_norm - ep_norm
        pose = pose.view(N * V, 6)
        pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm = \
            pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3], pose[:, 4], pose[:, 5]
        pitch, yaw, roll, s, tx, ty = \
            bfm_utils.denormalize_pose_params(pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm)
        sp, ep, _ = self.bfm_torch.denormalize_BFM_params(sp_norm, ep_norm, None, 'normal_manual')

        # Process vertices
        vert = self.bfm_torch.generate_vertices(sp, ep)
        vert[:, :, 2] -= 7.5e4
        if ap_norm is not None:
            # ap_norm = ap_norm.view(N * V, self.n_adap_para, 1) * 1e3
            ap_norm = self.adap_B_net.denormalize_ap_norm(ap_norm.view(N * V, self.n_adap_para, 1))
            delta_vert_ori = torch.bmm(self.adap_B, ap_norm).view(N * V, vert.shape[1], 3)
            delta_vert = delta_vert_ori.clone()
            # delta_vert[:, np.invert(self.bfm.face_region_mask), :] = 0.
            delta_vert_ori = delta_vert_ori.view(N, V, int(self.bfm.nver), 3)
            # delta_vert_ori[:, :, np.invert(self.bfm.face_region_mask), :] = 0.
            # delta_adap_bias = self.adap_bias.expand(N, V, vert.shape[1], 3).contiguous().view(N * V, vert.shape[1], 3)
            # delta_adap_bias[:, np.invert(self.bfm.face_region_mask), :] = 0.
            pre_adap_delta = self.pre_adap_delta.view(N * V, vert.shape[1], 3)
            # pre_adap_delta[:, np.invert(self.bfm.face_region_mask), :] = 0.
            vert = vert + delta_vert + pre_adap_delta# + delta_adap_bias
        else:
            delta_vert_ori = None
        angles = torch.stack([pitch, yaw, roll], dim=1)
        zeros = torch.zeros_like(tx)
        t = torch.stack([tx, ty, zeros], dim=1)
        vert = self.bfm_torch.transform(vert, s, angles, t)                 # dim: (N * V, nver, 3)

        return vert.view(N, V, int(self.bfm.nver), 3), delta_vert_ori

    def sample_per_vert_feat(self, feat, vert, H_img, W_img):
        """
        Project 3D vertices onto the feature map and sample feature vector for each 3D vertex
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: vert_feat: feature per vertex. (N, V, C, nver)
        """
        N, V, C, H, W = feat.shape
        feat = feat.view(N * V, C, H, W)
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Sample features
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img)
        grid = cam_opt.x_2d_normalize(H_img, W_img, vert_img[:, :, :2].clone()).view(N * V, int(self.bfm.nver), 1, 2)
        # vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros')
        grid_sample = MyGridSample.apply
        vert_feat = grid_sample(grid, feat)

        return vert_feat.view(N, V, C, int(self.bfm.nver))

    def compute_visibility_mask(self, vert):
        """
        Compute visibility mask of the given vertices of face mesh
        :param vert: (N, V, nver, 3)
        :return: vis_mask: visibility mask. (N, V, 1, nver)
        """
        N, V, _, _ = vert.shape
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Get visibility map
        # vis_mask = self.bfm_torch.backface_culling_cpu(vert, self.bfm.model['tri'])         # dim: (N * V, nver, 1)
        # full_face_vis_mask = torch.from_numpy(np.copy(vis_mask)).to(vert.device)            # dim: (N * V, nver, 1)
        # vis_mask = np.logical_and(vis_mask, self.bfm.face_region_mask[np.newaxis, ...])
        # vis_mask = torch.from_numpy(vis_mask).to(vert.device)                               # dim: (N * V, nver, 1)
        with torch.no_grad():
            vis_mask = self.bfm_torch.backface_culling(vert, self.bfm)                      # dim: (N * V, nver, 1)
            full_face_vis_mask = vis_mask.clone()
            vis_mask = vis_mask.byte() * self.bfm_torch.face_region_mask.byte()

        return vis_mask.byte().view(N, V, 1, int(self.bfm.nver)),\
               full_face_vis_mask.byte().view(N, V, 1, int(self.bfm.nver))

    def feature_metric_loss(self, feat, vis_mask):
        N, V, C, nver = feat.shape
        loss = 0
        abs_residuals_mean = [[] for i in range(V)]
        for i in range(V - 1):
            for j in range(i + 1, V):
                # Compute loss
                cur_loss = F.mse_loss(feat[:, i, :, :], feat[:, j, :, :], reduction='none')     # dim: (N, C, nver)
                cur_loss_sum = torch.sum(cur_loss, dim=1, keepdim=True)                         # dim: (N, 1, nver)
                # cur_loss_sum = F.cosine_similarity(feat[:, i, :, :], feat[:, j, :, :], dim=1).view(N, 1, nver)
                mask = vis_mask[:, i, :, :] * vis_mask[:, j, :, :]
                cur_loss_masked = torch.masked_select(cur_loss_sum, mask.bool())
                err = torch.mean(cur_loss_masked)
                loss += err

                # Compute abs residual
                abs_residual = torch.abs(feat[:, i, :, :] - feat[:, j, :, :])           # dim: (N, C, nver)
                abs_residual_sum = torch.sum(abs_residual * mask.float(), dim=2)        # dim: (N, C)
                abs_residual_mean = abs_residual_sum / torch.sum(mask.float(), dim=2)   # dim: (N, C)
                abs_residuals_mean[i].append(abs_residual_mean)
                abs_residuals_mean[j].append(abs_residual_mean)
        abs_residuals_mean = [torch.mean(torch.stack(abs_residuals_mean[i], dim=0), dim=0) for i in range(V)]
        abs_residuals = torch.stack(abs_residuals_mean, dim=1)                          # dim: (N, V, C)

        return loss / (V * (V - 1.0) / 2.0), abs_residuals

    def landmark_loss(self, vert, kpts_gt, kpts_3d_gt, full_face_vis_mask, vis_mask, H_img, W_img):
        N, V, nver, _ = vert.shape
        vert = vert.view(N * V, nver, 3)
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img).view(N, V, nver, 3)
        kpts = vert_img[:, :, self.bfm.kpt_ind, :]                                              # dim: (N, V, 68, 3)

        # Get fixed landmark mask
        kpts_vis_mask = full_face_vis_mask[:, :, 0, self.bfm.kpt_ind]                           # dim: (N, V, 68)
        invar_idx = np.concatenate([np.arange(6, 11), np.arange(17, 68)])
        kpts_fix_mask = torch.zeros_like(kpts_vis_mask)                                         # dim: (N, V, 68)
        kpts_fix_mask[:, :, invar_idx] = 1
        kpts_fix_mask = torch.min(kpts_fix_mask + kpts_vis_mask, torch.ones_like(kpts_fix_mask))
        eyebrows_idx = np.arange(17, 27)

        # Compute loss on fixed landmarks
        loss = F.mse_loss(kpts[:, :, :, :2], kpts_gt[:, :, :, :2], reduction='none')            # dim: (N, V, 68, 2)
        loss_sum = torch.sum(loss, dim=3) * kpts_fix_mask.float()                               # dim: (N, V, 68)
        # loss_sum[:, :, eyebrows_idx] = 0.0
        err = torch.mean(loss_sum)

        # Compute abs residual
        fix_abs_residual = torch.abs(kpts[:, :, :, :2] - kpts_gt[:, :, :, :2])                  # dim: (N, V, 68, 2)
        fix_abs_residual_mean = torch.mean(fix_abs_residual * kpts_fix_mask.unsqueeze(3).float(), dim=2)# dim: (N, V, 2)

        # Select closest vert to dynamic landmarks
        invis_factor = (1 - vis_mask) * 1e6                                                     # dim: (N, V, 1, nver)
        vert_to_kpts_dist = torch.norm(
            vert_img[:, :, :, :2].detach().view(N, V, nver, 1, 2) - kpts_gt.view(N, V, 1, 68, 2),
            p=2, dim=-1)                                                                        # dim: (N, V, nver, 68)
        vert_to_kpts_dist = vert_to_kpts_dist + invis_factor.view(N, V, nver, 1).float()
        cls_vert_dix = torch.argmin(vert_to_kpts_dist, dim=2)                                   # dim: (N, V, 68)
        cls_vert = torch.gather(vert_img[:, :, :, :2].view(N, V, nver, 1, 2).expand(N, V, nver, 68, 2),
                                index=cls_vert_dix.view(N, V, 1, 68, 1).expand(N, V, 1, 68, 2),
                                dim=2).view(N, V, 68, 2)                                        # dim: (N, V, 68, 2)

        # Compute loss on dynamic landmarks
        loss = F.mse_loss(cls_vert, kpts_gt, reduction='none')                                  # dim: (N, V, 68, 2)
        loss_sum = torch.sum(loss, dim=3) * (1.0 - kpts_fix_mask.float())                       # dim: (N, V, 68)
        err += torch.mean(loss_sum)

        # Compute abs residual
        dyn_abs_residual = torch.abs(cls_vert[:, :, :, :] - kpts_gt[:, :, :, :])                # dim: (N, V, 68, 2)
        dyn_abs_residual_mean = torch.mean(dyn_abs_residual * (1.0 - kpts_fix_mask.float()).unsqueeze(3), dim=2)# dim: (N, V, 2)
        abs_residuals = fix_abs_residual_mean + dyn_abs_residual_mean                           # dim: (N, V, 2)
        abs_residuals[:, :, 0] = abs_residuals[:, :, 0] / (W_img - 1.)
        abs_residuals[:, :, 1] = abs_residuals[:, :, 1] / (H_img - 1.)

        return err, abs_residuals

    def reg_loss(self, sp_norm, ep_norm):
        abs_residual_sp = (sp_norm * sp_norm)[:, :80, :].squeeze(2)
        abs_residual_ep = torch.mean(ep_norm * ep_norm, dim=1).squeeze(2)
        sp_reg = torch.mean(abs_residual_sp)
        ep_reg = torch.mean(abs_residual_ep)
        return sp_reg, abs_residual_sp, abs_residual_ep

    def gradient_descent(self, images, feat, kpts, kpts_3d, vert, H_img, W_img, sp_norm, ep_norm, pose, ap_norm, vis_mask,
                         full_face_vis_mask, step_net, hard_opt, step_s, step_b):
        """
        Update params with 1st order optimization
        :param images: (N, V, C_img, H_img, W_img)
        :param feat: (N, V, C, H, W)
        :param kpts: (N, V, 68, 2)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param sp_norm:
        :param ep_norm:
        :param pose:
        :param ap_norm:
        :param vis_mask: visibility mask of tight face region. np.array (N, V, 1, nver)
        :param full_face_vis_mask: visibility mask of full BFM mesh. np.array (N, V, 1, nver)
        :return:
        """
        N, V, C, _, _ = feat.shape
        with torch.enable_grad():
            vert_feat = self.sample_per_vert_feat(feat, vert, H_img, W_img)             # dim: (N, V, C, nver)
            feat_loss, feat_abs_res = self.feature_metric_loss(vert_feat, vis_mask)
            # print(feat_loss)
            # print(torch.norm(feat_abs_res, dim=1))
            land_loss, land_abs_res = self.landmark_loss(vert, kpts, kpts_3d, full_face_vis_mask, vis_mask, H_img, W_img)
            # print(torch.norm(land_abs_res, dim=1))
            # reg_loss, sp_abs_res, ep_abs_res = self.reg_loss(sp_norm, ep_norm)
            loss = 0.25 * feat_loss + 0.025 * land_loss #if feat is not None else 0.025 * land_loss + 0.01 * reg_loss
            loss = loss * N

        # Compute gradient and update params
        # if feat is None:
        if ap_norm is not None:
            sp_norm_grad, ep_norm_grad, pose_grad, ap_norm_grad = torch.autograd.grad(loss, [sp_norm, ep_norm, pose, ap_norm], create_graph=self.training)
        else:
            sp_norm_grad, ep_norm_grad, pose_grad = torch.autograd.grad(loss, [sp_norm, ep_norm, pose], create_graph=self.training)
        # else:
        #     sp_norm_grad, ep_norm_grad = torch.autograd.grad(feat_loss, [sp_norm, ep_norm], create_graph=True)
        if ap_norm is not None:
            N = vert.shape[0]
            if hard_opt:
                pass
            else:
                feat_abs_res = torch.mean(feat_abs_res, dim=1)          # (N, C)
                land_abs_res = torch.mean(land_abs_res, dim=1)          # (N, 2)
                (sp_step_size, ep_step_size, pose_step_size, ap_step_size), raw_step_size = \
                    step_net((feat_abs_res, land_abs_res), step_s, step_b)
                sp_step_size = sp_step_size.view(N, 1, 1)
                ep_step_size = ep_step_size.view(N, 1, 1, 1)
                pose_step_size = pose_step_size.view(N, 1, 1)
                ap_step_size = ap_step_size.view(N, 1, 1, 1)
                # print(torch.max(sp_step_size), torch.max(ep_step_size), torch.max(pose_step_size), torch.max(ap_step_size))
                # (sp_step_size, ep_step_size, pose_step_size, ap_step_size), raw_step_size = \
                #     step_net((feat_abs_res.view(N * V, C), land_abs_res.view(N * V, 2)), step_s, step_b)#(2., 2., 2.), (-0.5, -0.5, -2.5))
                # sp_step_size = torch.mean(sp_step_size.view(N, V, 1), dim=1).view(N, 1, 1) \
                #                .expand(N, 1, self.bfm.n_shape_para // 1).contiguous().view(N, self.bfm.n_shape_para, 1)
                # ep_step_size = ep_step_size.view(N, V, 1, 1).expand(N, V, 1, self.bfm.n_exp_para // 1) \
                #                .contiguous().view(N, V, self.bfm.n_exp_para, 1)
                # pose_step_size = pose_step_size.view(N, V, 1)
                # ap_step_size = ap_step_size.view(N, V, 1, 1).expand(N, V, 1, self.n_adap_para // 1) \
                #                .contiguous().view(N, V, self.n_adap_para, 1)
                # print(torch.min(sp_step_size), torch.min(ep_step_size), torch.min(pose_step_size), torch.min(ap_step_size))
                # print(torch.max(sp_step_size), torch.max(ep_step_size), torch.max(pose_step_size), torch.max(ap_step_size))
        else:
            N = vert.shape[0]
            if hard_opt:
                (step_size,), raw_step_size = \
                    step_net((feat_abs_res, land_abs_res, sp_abs_res, ep_abs_res), (5.,), (-3.,))
                sp_step_size = step_size.view(N, 1, 1)
                ep_step_size = step_size.view(N, 1, 1, 1)
                pose_step_size = step_size.view(N, 1, 1)
                print(step_size.view(N,))
            else:
                (sp_step_size, ep_step_size, pose_step_size), raw_step_size = \
                    step_net((feat_abs_res.view(N * V, C), land_abs_res.view(N * V, 2)), step_s, step_b)#(2., 2., 2.), (0, 0, -2.5))
                sp_step_size = torch.mean(sp_step_size.view(N, V, 16), dim=1).view(N, 16, 1) \
                               .expand(N, 16, self.bfm.n_shape_para // 16).contiguous().view(N, self.bfm.n_shape_para, 1)
                ep_step_size = ep_step_size.view(N, V, 16, 1).expand(N, V, 16, self.bfm.n_exp_para // 16) \
                               .contiguous().view(N, V, self.bfm.n_exp_para, 1)
                pose_step_size = pose_step_size.view(N, V, 1)
                print(torch.max(sp_step_size), torch.max(ep_step_size), torch.max(pose_step_size))
            # # sp_norm_grad_in = sp_norm_grad[:, :80, :].view(N, 80)
            # # ep_norm_grad_in = torch.mean(ep_norm_grad, dim=1).view(N, self.bfm.n_exp_para)
            # # pose_grad_in = torch.mean(pose_grad, dim=1).view(N, 6)
            # (sp_step_size, ep_step_size, pose_step_size), raw_step_size = \
            #     step_net((feat_abs_res, land_abs_res, sp_abs_res, ep_abs_res), (5., 5., 5.), (0., 0., -2.))
            # # (sp_step_size, ep_step_size, pose_step_size), raw_step_size = \
            # #     step_net((feat_abs_res, land_abs_res, sp_norm_grad_in, ep_norm_grad_in, pose_grad_in), (3., 3., 3.), (-2., -2., -3.))
            # sp_step_size = sp_step_size.view(N, self.bfm.n_shape_para, 1)
            # ep_step_size = ep_step_size.view(N, 1, self.bfm.n_exp_para, 1)
            # pose_step_size = pose_step_size.view(N, 1, 6)
            # # sp_step_size = sp_step_size.view(N, 1, 1)
            # # ep_step_size = ep_step_size.view(N, 1, 1, 1)
            # # pose_step_size = pose_step_size.view(N, 1, 1)
            # print(torch.max(sp_step_size * sp_norm_grad), torch.max(ep_step_size * ep_norm_grad), torch.max(pose_step_size * pose_grad))
            # # sp_step_size = self.opt_step_size * 100. * step_size_factor
            # # ep_step_size = self.opt_step_size * 500. * step_size_factor
            # # pose_step_size = self.opt_step_size * step_size_factor
        # else:
        #     N = vert.shape[0]
        #     # C = self.step_net.in_nchs[0]
        #     # feat_abs_res = torch.zeros((N, C), dtype=torch.float).cuda()
        #     (sp_step_size, ep_step_size, pose_step_size), raw_step_size = \
        #         step_net((land_abs_res, sp_abs_res, ep_abs_res), (5., 5., 5.), (0., 0., -2.))
        #     sp_step_size = sp_step_size.view(N, self.bfm.n_shape_para, 1)
        #     ep_step_size = ep_step_size.view(N, 1, self.bfm.n_exp_para, 1)
        #     pose_step_size = pose_step_size.view(N, 1, 6)
        #     print(torch.max(sp_step_size), torch.max(ep_step_size), torch.max(pose_step_size))
        #     # sp_step_size = self.opt_step_size * 100. * step_size_factor
        #     # ep_step_size = self.opt_step_size * 500. * step_size_factor
        #     # pose_step_size = self.opt_step_size * step_size_factor
        with torch.enable_grad():
            sp_norm = sp_norm - sp_step_size * sp_norm_grad
            ep_norm = ep_norm - ep_step_size * ep_norm_grad
            # if feat is None:
            pose = pose - pose_step_size * pose_grad
            if ap_norm is not None:
                ap_norm = ap_norm - ap_step_size * ap_norm_grad

        new_vert, delta_vert = self.compute_vert_from_params(sp_norm, ep_norm, pose, ap_norm)
        N, V, nver, _ = new_vert.shape
        new_vert_img = self.bfm_torch.to_image(new_vert.view(N * V, nver, 3), H_img, W_img).view(N, V, nver, 3)
        new_colors = self.sample_per_vert_feat(images, vert.detach(), H_img, W_img)
        vis_mask, full_face_vis_mask = self.compute_visibility_mask(new_vert)       # dim: (N, V, 1, nver)

        return sp_norm, ep_norm, pose, ap_norm, new_vert, new_vert_img, new_colors, vis_mask, full_face_vis_mask, \
               raw_step_size, delta_vert

    def forward(self, rgb_feats, kpts, kpts_3d, sp_norm, ep_norm, pose, images, ori_images, only_regr):
        """
        Main loop of the NRMVS optimization
        :param rgb_feats: image feautres extracted by DRN. [(N * V, c, h, w)]
        :param kpts: 2D lamdmarks. (N, V, 68, 2)
        :param sp_norm: (N, n_shape_para, 1)
        :param ep_norm: (N, V, n_exp_para, 1)
        :param pose: (N, V, 6)
        :param images: (N, V, C, H, W)
        :param ori_images: (N, V, C, H, W)
        :return:
        """
        N, V, _ = pose.shape
        _, _, C_img, H_img, W_img = ori_images.shape

        # Set initial params & outputs
        # sp_norm = sp_norm.detach()
        # ep_norm = ep_norm.detach()
        # pose = pose.detach()
        # sp_norm.requires_grad = True
        # ep_norm.requires_grad = True
        # pose.requires_grad = True
        verts = []
        verts_img = []
        colors_list = []
        vis_masks = []
        full_face_vis_masks = []
        raw_step_sizes = []
        ori_colors_list = []
        adap_B_uv_list = []
        delta_vert_list = []

        # Generate initial vertices (mesh)
        vert, _ = self.compute_vert_from_params(sp_norm, ep_norm, pose, None)
        nver = vert.shape[2]
        vert_img = self.bfm_torch.to_image(vert.view(N * V, nver, 3), H_img, W_img).view(N, V, nver, 3)
        colors = self.sample_per_vert_feat(images, vert, H_img, W_img)
        vis_mask, full_face_vis_mask = self.compute_visibility_mask(vert)       # dim: (N, V, 1, nver)
        verts.append(vert)
        verts_img.append(vert_img)
        colors_list.append(colors)
        vis_masks.append(vis_mask)
        full_face_vis_masks.append(full_face_vis_mask)
        ori_colors_list.append(self.sample_per_vert_feat(ori_images, vert.detach(), H_img, W_img))

        if only_regr:
            return verts, verts_img, vis_masks, full_face_vis_masks, ori_colors_list, colors_list, raw_step_sizes

        sp_norm = sp_norm.detach()
        ep_norm = ep_norm.detach()
        pose = pose.detach()
        sp_norm.requires_grad = True
        ep_norm.requires_grad = True
        pose.requires_grad = True
        vert, _ = self.compute_vert_from_params(sp_norm, ep_norm, pose, None)
        nver = vert.shape[2]
        vis_mask, full_face_vis_mask = self.compute_visibility_mask(vert)       # dim: (N, V, 1, nver)

        # Get feature pyramid
        rgb_feats = self.rgb_net(images.view(N * V, C_img, H_img, W_img))           # dim: [(N * V, c, h, w)]
        # fpn_feats = self.fpn(rgb_feats[0:5])
        fpn_feats = self.fpn(rgb_feats[0:6])
        # fpn_feats = rgb_feats[1:5]
        basis_fpn_feats = self.basis_fpn(rgb_feats[0:6])

        # Optimization level
        # cur_level_verts = []
        # cur_level_verts_img = []
        # cur_level_colors_list = []
        # cur_level_vis_masks = []
        # cur_level_full_face_vis_masks = []
        # cur_level_raw_step_sizes = []
        # for i in range(3):
        #     sp_norm, ep_norm, pose, delta_vert, vert, vert_img, colors, vis_mask, full_face_vis_mask, raw_step_size = \
        #         self.gradient_descent(images, None, kpts, kpts_3d, vert, H_img, W_img, sp_norm, ep_norm, pose, None,
        #                               vis_mask, full_face_vis_mask, 1., self.step_net2)
        #     cur_level_verts.append(vert)
        #     cur_level_verts_img.append(vert_img)
        #     cur_level_colors_list.append(colors)
        #     cur_level_vis_masks.append(vis_mask)
        #     cur_level_full_face_vis_masks.append(full_face_vis_mask)
        #     cur_level_raw_step_sizes.append(raw_step_size)
        # ori_colors_list.append(self.sample_per_vert_feat(ori_images, vert.detach(), H_img, W_img))
        # verts.append(cur_level_verts)
        # verts_img.append(cur_level_verts_img)
        # colors_list.append(cur_level_colors_list)
        # vis_masks.append(cur_level_vis_masks)
        # full_face_vis_masks.append(cur_level_full_face_vis_masks)
        # raw_step_sizes.append(cur_level_raw_step_sizes)

        # delta_vert = torch.zeros_like(vert, requires_grad=True)
        # vert = vert + delta_vert
        ap_norm = None

        step_s = [(2., 2., 2., 4.), (4., 4., 4., 4.), (4., 4., 4., 4.)]
        step_b = [(0., 0., -2.5, 0.), (0., 0., -2.5, 0.), (0., 0., -2.5, 0.)]
        for l in range(2, len(fpn_feats)):
            feat = fpn_feats[l]
            basis_feat = basis_fpn_feats[l]
            _, C, H, W = feat.shape
            feat = feat.view(N, V, C, H, W)
            basis_feat = basis_feat.view(N, V, C, H, W)
            cur_level_verts = []
            cur_level_verts_img = []
            cur_level_colors_list = []
            cur_level_vis_masks = []
            cur_level_full_face_vis_masks = []
            cur_level_raw_step_sizes = []
            cur_level_delta_vert_list = []

            # Adaptive Basis Generation
            if ap_norm is None:
                self.pre_adap_delta = torch.zeros_like(vert)
            else:
                self.pre_adap_delta = self.pre_adap_delta + delta_vert.detach()
            adap_B_net = getattr(self, 'adap_B_net' + str(l))
            self.adap_B_net = adap_B_net
            self.n_adap_para = adap_B_net.n_adap_para
            ap_norm = torch.zeros((N, V, self.n_adap_para, 1), requires_grad=True, device=images.device)
            self.adap_B, self.adap_bias, adap_B_uv = adap_B_net(basis_feat, vert.detach(), vis_mask, H_img, W_img)          # (N, nver * 3, n_adap_para)
            adap_B_uv_list.append(self.adap_B)
            self.adap_B = self.adap_B.view(N, 1, int(self.bfm.nver) * 3, self.n_adap_para) \
                                     .expand(N, V, int(self.bfm.nver) * 3, self.n_adap_para) \
                                     .contiguous().view(N * V, int(self.bfm.nver) * 3, self.n_adap_para)
            with torch.enable_grad():
                sp_norm = sp_norm.detach()
                ep_norm = ep_norm.detach()
                pose = pose.detach()
                sp_norm.requires_grad = True
                ep_norm.requires_grad = True
                pose.requires_grad = True
            vert, delta_vert = self.compute_vert_from_params(sp_norm, ep_norm, pose, ap_norm)
            vis_mask, full_face_vis_mask = self.compute_visibility_mask(vert)           # dim: (N, V, 1, nver)
            # Optimization iter
            for i in range(3):
                # Compute updated params
                sp_norm, ep_norm, pose, ap_norm, vert, vert_img, colors, vis_mask, full_face_vis_mask, raw_step_size, \
                    delta_vert = self.gradient_descent(images, feat, kpts, kpts_3d, vert, H_img, W_img,
                                                       sp_norm, ep_norm, pose, ap_norm, vis_mask, full_face_vis_mask,
                                                       getattr(self, 'step_net' + str(l)), False,
                                                       step_s[l - 2], step_b[l - 2])
                cur_level_verts.append(vert)
                cur_level_verts_img.append(vert_img)
                cur_level_colors_list.append(colors)
                cur_level_vis_masks.append(vis_mask)
                cur_level_full_face_vis_masks.append(full_face_vis_mask)
                cur_level_raw_step_sizes.append(raw_step_size)
                cur_level_delta_vert_list.append(delta_vert)

            ori_colors_list.append(self.sample_per_vert_feat(ori_images, vert.detach(), H_img, W_img))
            verts.append(cur_level_verts)
            verts_img.append(cur_level_verts_img)
            colors_list.append(cur_level_colors_list)
            vis_masks.append(cur_level_vis_masks)
            full_face_vis_masks.append(cur_level_full_face_vis_masks)
            raw_step_sizes.append(cur_level_raw_step_sizes)
            delta_vert_list.append(cur_level_delta_vert_list)

        return verts, verts_img, vis_masks, full_face_vis_masks, ori_colors_list, colors_list, raw_step_sizes, \
               adap_B_uv_list, delta_vert_list


class FNRMVSNet(BaseNet):

    def __init__(self, n_shape_para=199, n_exp_para=29, opt_step_size=1e-5,
                 MM_base_dir='./external/face3d/examples/Data/BFM'):
        super(FNRMVSNet, self).__init__()
        self.regressor = RegressionNet(n_shape_para=n_shape_para, n_exp_para=n_exp_para)
        # self.bfm_dec = BFMDecoder(MM_base_dir=MM_base_dir)
        self.opt_layer = NRMVSOptimization(opt_step_size=opt_step_size, MM_base_dir=MM_base_dir)

    def save_net_def(self, dir):
        super(FNRMVSNet, self).save_net_def(dir)
        shutil.copy(os.path.realpath(__file__), dir)

    def cuda(self, device=None):
        super(FNRMVSNet, self).cuda(device)
        self.regressor.cuda(device)
        # self.bfm_dec.cuda(device)
        self.opt_layer.cuda(device)

    def train(self, mode=True):
        super(FNRMVSNet, self).train(mode)
        self.regressor.eval()

    def parameters_wo_s_fc(self):
        return itertools.chain(
            self.regressor.rgb_net.parameters(),
            self.regressor.shared_branch.parameters(),
            self.regressor.iden_fc.parameters(),
            self.regressor.per_view_branch.parameters(),
            self.regressor.exp_pose_fc.parameters(),
            self.regressor.exp_fc.parameters(),
            self.regressor.angles_fc.parameters(),
            self.regressor.t_fc.parameters()
        )

    def parameters_cnns(self):
        return itertools.chain(
            self.opt_layer.rgb_net.parameters(),
            self.opt_layer.fpn.parameters(),
            self.opt_layer.adap_B_net2.parameters(),
            self.opt_layer.adap_B_net3.parameters()
        )

    def parameters_stepnets(self):
        return itertools.chain(
            self.opt_layer.step_net2.parameters(),
            self.opt_layer.step_net3.parameters()
        )

    def forward(self, images, ori_images, kpts, kpts_3d, only_regr):
        if only_regr:
            sp_norm, ep_norm, pose, rgb_feats = self.regressor.forward(images)
        else:
            with torch.no_grad():
                sp_norm, ep_norm, pose, rgb_feats = self.regressor.forward(images)
        #
        # colors, vis_mask, full_face_vis_mask, vert_img, ori_colors = self.bfm_dec.forward(images, sp_norm, ep_norm, pose, ori_images)
        #
        # opt_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, ori_colors_list, colors_list = \
        #     self.opt_layer.forward(rgb_feats, kpts, kpts_3d, sp_norm, ep_norm, pose, images, ori_images)
        #
        # return sp_norm, ep_norm, pose, colors, vis_mask, full_face_vis_mask, vert_img, ori_colors,\
        #        opt_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, ori_colors_list, colors_list

        N, V, _, _, _ = images.shape
        sp_norm = torch.zeros((N, self.opt_layer.bfm.n_shape_para, 1), device=images.device)
        ep_norm = torch.zeros((N, V, self.opt_layer.bfm.n_exp_para, 1), device=images.device)
        # pose = torch.zeros((N, V, 6), device=images.device)
        opt_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, ori_colors_list, colors_list, raw_step_sizes, \
            adap_B_uv_list, delta_vert_list = \
            self.opt_layer.forward(None, kpts, kpts_3d, sp_norm, ep_norm, pose, images, ori_images, only_regr)

        return sp_norm, ep_norm, pose, None, None, None, None, None,\
               opt_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, ori_colors_list, colors_list, \
               raw_step_sizes, adap_B_uv_list, delta_vert_list


if __name__ == '__main__':
    with torch.cuda.device(0):
        model = RegressionNet()
        model.cuda()
        rand_input = torch.rand(2, 4, 3, 256, 256).cuda()
        sp_norm, ep_norm, pose, _ = model.forward(rand_input)
        print(sp_norm.shape, ep_norm.shape, pose.shape)