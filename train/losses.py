import torch
import torch.nn.functional as F
import numpy as np

# External libs
import external.face3d.face3d.mesh as mesh

# Internal libs
import data.utils as utils


class LossFunction(object):

    def __init__(self, mean_loss=True):
        super(LossFunction, self).__init__()
        self.mean_loss = mean_loss

    def dense_aligned_vert_loss(self, vert, vert_bfm_gt, face_region_mask, root=False):
        N, V, nver, _ = vert.shape
        face_region_idx = np.arange(0, face_region_mask.shape[0], 1).astype(np.int)[face_region_mask.ravel()]

        # Compute dense alignment
        # s_nps = []
        # R_nps = []
        # t_nps = []
        # for i in range(N):
        #     cur_s_nps = []
        #     cur_R_nps = []
        #     cur_t_nps = []
        #     for j in range(V):
        #         # Dense alignment
        #         d, Z, tform = utils.procrustes_pytorch(
        #             vert_bfm_gt[i, j, face_region_idx, :].clone(),
        #             vert[i, j, face_region_idx, :].detach().clone(),
        #             scaling=True, reflection='best')
        #         s = tform['scale']
        #         R = tform['rotation']
        #         t = tform['translation']
        #         cur_s_nps.append(s)
        #         cur_R_nps.append(R)
        #         cur_t_nps.append(t)
        #     s_nps.append(torch.stack(cur_s_nps, dim=0))
        #     R_nps.append(torch.stack(cur_R_nps, dim=0))
        #     t_nps.append(torch.stack(cur_t_nps, dim=0))
        # s = torch.stack(s_nps, dim=0).to(vert.device).view(N, V, 1, 1)
        # R = torch.stack(R_nps, dim=0).to(vert.device).view(N, V, 3, 3)
        # t = torch.stack(t_nps, dim=0).to(vert.device).view(N, V, 1, 3)
        X = vert.detach().clone().view(N * V, nver, 3)[:, face_region_idx, :]
        X_gt = vert_bfm_gt.clone().view(N * V, nver, 3)[:, face_region_idx, :]
        d, Z, tform = utils.batched_procrustes_pytorch(X_gt, X, scaling=True, reflection=False)
        s = tform['scale'].view(N, V, 1, 1)
        R = tform['rotation'].view(N, V, 3, 3)
        t = tform['translation'].view(N, V, 1, 3)
        vert = s * torch.bmm(vert.view(N * V, nver, 3), R.view(N * V, 3, 3)).view(N, V, nver, 3) + t

        # Compute loss
        vert_face = vert[:, :, face_region_idx, :]
        vert_face_gt = vert_bfm_gt[:, :, face_region_idx, :]
        loss = F.mse_loss(vert_face, vert_face_gt, reduction='none')  # dim: (N, V, M, 3)
        loss = torch.sum(loss, dim=3)  # dim: (N, V, M)
        loss = torch.sqrt(loss) if root else loss
        loss = torch.mean(loss, dim=2)  # dim: (N, V)
        return loss

    def depth_aligned_vert_loss(self, vert, vert_bfm_gt, face_region_mask, root=False):
        face_region_idx = np.arange(0, face_region_mask.shape[0], 1).astype(np.int)[face_region_mask.ravel()]

        # Align depth
        vert_face = vert[:, :, face_region_idx, :].detach()
        vert_bfm_face_gt = vert_bfm_gt[:, :, face_region_idx, :]
        N, V, M, _ = vert_face.shape
        delta_z = torch.mean(vert_face[:, :, :, 2] - vert_bfm_face_gt[:, :, :, 2], dim=2).view(N, V, 1, 1)
        vert_gt = torch.cat([vert_bfm_gt[:, :, :, :2], vert_bfm_gt[:, :, :, 2:] + delta_z], dim=3)

        # Compute loss
        vert_face = vert[:, :, face_region_idx, :]
        vert_face_gt = vert_gt[:, :, face_region_idx, :]
        loss = F.mse_loss(vert_face, vert_face_gt, reduction='none')                # dim: (N, V, M, 3)
        loss = torch.sum(loss, dim=3)                                               # dim: (N, V, M)
        loss = torch.mean(loss, dim=2)                                              # dim: (N, V)
        return loss

    def vert_loss(self, opt_verts, vert_bfm_gt, face_region_mask, align_mode='depth'):
        if align_mode == 'depth':
            aligned_vert_loss = self.depth_aligned_vert_loss
        elif align_mode == 'dense':
            aligned_vert_loss = self.dense_aligned_vert_loss
        reg_err = aligned_vert_loss(opt_verts[0].detach(), vert_bfm_gt, face_region_mask, root=True)
        land_opt_err = aligned_vert_loss(opt_verts[1][-1].detach(), vert_bfm_gt, face_region_mask, root=True)
        feat_opt_err = aligned_vert_loss(opt_verts[-1][-1].detach(), vert_bfm_gt, face_region_mask, root=True)
        loss = 0.
        for opt_stage in range(1, len(opt_verts)):
            for opt_itr in range(len(opt_verts[opt_stage])):
                cur_loss = aligned_vert_loss(opt_verts[opt_stage][opt_itr], vert_bfm_gt, face_region_mask, root=False)
                loss += torch.mean(cur_loss)
        return loss, reg_err.detach(), land_opt_err.detach(), feat_opt_err.detach()

    def land3d_loss(self, vert_img, kpts_gt, kpts_3d_gt, bfm, full_face_vis_mask, vis_mask):
        N, V, nver, _ = vert_img.shape
        kpts_3d = vert_img[:, :, bfm.kpt_ind, :]                            # dim: (N, V, 68, 3)

        # Get fixed landmark mask
        kpts_vis_mask = full_face_vis_mask[:, :, 0, bfm.kpt_ind]  # dim: (N, V, 68)
        invar_idx = np.concatenate([np.arange(6, 11), np.arange(17, 68)])
        kpts_fix_mask = torch.zeros_like(kpts_vis_mask)  # dim: (N, V, 68)
        kpts_fix_mask[:, :, invar_idx] = 1
        kpts_fix_mask = torch.min(kpts_fix_mask + kpts_vis_mask, torch.ones_like(kpts_fix_mask))
        low_weight_idx = np.arange(27, 48)
        high_weight_idx = np.concatenate([np.arange(0, 27), np.arange(48, 68)])

        # Compute loss on 3D landmarks
        low_weight_loss = F.mse_loss(kpts_3d[:, :, low_weight_idx, :2],
                                     kpts_3d_gt[:, :, low_weight_idx, :2], reduction='none')        # dim: (N, V, 51, 2)
        high_weight_loss = F.mse_loss(kpts_3d[:, :, high_weight_idx, :2],
                                  kpts_3d_gt[:, :, high_weight_idx, :2], reduction='none')              # dim: (N, V, 17, 2)

        low_weight_loss_sum = torch.sum(low_weight_loss, dim=3)                                     # dim: (N, V, 51)
        high_weight_loss_sum = torch.sum(high_weight_loss, dim=3)                                           # dim: (N, V, 17)
        if self.mean_loss:
            err = torch.mean(low_weight_loss_sum) + 10. * torch.mean(high_weight_loss_sum)
        else:
            err = torch.sum(low_weight_loss_sum) + 10. * torch.sum(high_weight_loss_sum)

        # Select closest vert to dynamic landmarks
        invis_factor = (1 - vis_mask) * 1e6                                         # dim: (N, V, 1, nver)
        vert_to_kpts_dist = torch.norm(vert_img[:, :, :, :2].detach().view(N, V, nver, 1, 2) - kpts_gt.view(N, V, 1, 68, 2),
                                       p=2, dim=-1)                                 # dim: (N, V, nver, 68)
        vert_to_kpts_dist = vert_to_kpts_dist + invis_factor.view(N, V, nver, 1).float()
        cls_vert_dix = torch.argmin(vert_to_kpts_dist, dim=2)                       # dim: (N, V, 68)
        cls_vert = torch.gather(vert_img[:, :, :, :2].view(N, V, nver, 1, 2).expand(N, V, nver, 68, 2),
                                index=cls_vert_dix.view(N, V, 1, 68, 1).expand(N, V, 1, 68, 2),
                                dim=2).view(N, V, 68, 2)                            # dim: (N, V, 68, 2)

        # Compute loss on dynamic landmarks
        loss = F.mse_loss(cls_vert, kpts_gt, reduction='none')                      # dim: (N, V, 68, 2)
        loss_sum = torch.sum(loss, dim=3) * (1.0 - kpts_fix_mask.float())           # dim: (N, V, 68)
        err += torch.mean(loss_sum) if self.mean_loss else torch.sum(loss_sum)

        return err

    def multi_land3d_loss(self, opt_verts_img, kpts_gt, kpts_3d_gt, bfm, opt_full_face_vis_masks, opt_vis_masks):
        loss = 0.
        for opt_stage in range(1, len(opt_verts_img)):
            for opt_itr in range(len(opt_verts_img[opt_stage])):
                cur_loss = self.land3d_loss(opt_verts_img[opt_stage][opt_itr], kpts_gt, kpts_3d_gt, bfm,
                                            opt_full_face_vis_masks[opt_stage][opt_itr], opt_vis_masks[opt_stage][opt_itr])
                loss += torch.mean(cur_loss)
        return loss

    def normal_loss(self, vert, normal_bfm_gt, vert_bfm_gt, bfm):
        N, V, nver, _ = vert.shape
        face_region_idx = np.arange(0, bfm.face_region_mask.shape[0], 1).astype(np.int)[bfm.face_region_mask.ravel()]

        # Compute dense alignment
        # s_nps = []
        # R_nps = []
        # t_nps = []
        # for i in range(N):
        #     cur_s_nps = []
        #     cur_R_nps = []
        #     cur_t_nps = []
        #     for j in range(V):
        #         # Dense alignment
        #         d, Z, tform = utils.procrustes_pytorch(
        #             vert_bfm_gt[i, j, face_region_idx, :].clone(),
        #             vert[i, j, face_region_idx, :].detach().clone(),
        #             scaling=True, reflection='best')
        #         s = tform['scale']
        #         R = tform['rotation']
        #         t = tform['translation']
        #         cur_s_nps.append(s)
        #         cur_R_nps.append(R)
        #         cur_t_nps.append(t)
        #     s_nps.append(torch.stack(cur_s_nps, dim=0))
        #     R_nps.append(torch.stack(cur_R_nps, dim=0))
        #     t_nps.append(torch.stack(cur_t_nps, dim=0))
        # s = torch.stack(s_nps, dim=0).to(vert.device).view(N, V, 1, 1)
        # R = torch.stack(R_nps, dim=0).to(vert.device).view(N, V, 3, 3)
        # t = torch.stack(t_nps, dim=0).to(vert.device).view(N, V, 1, 3)
        X = vert.detach().clone().view(N * V, nver, 3)[:, face_region_idx, :]
        X_gt = vert_bfm_gt.clone().view(N * V, nver, 3)[:, face_region_idx, :]
        d, Z, tform = utils.batched_procrustes_pytorch(X_gt, X, scaling=True, reflection=False)
        s = tform['scale'].view(N, V, 1, 1)
        R = tform['rotation'].view(N, V, 3, 3)
        t = tform['translation'].view(N, V, 1, 3)
        vert = s * torch.bmm(vert.view(N * V, nver, 3), R.view(N * V, 3, 3)).view(N, V, nver, 3) + t

        # Compute normal loss
        pt0 = vert[:, :, bfm.model['tri'][:, 2], :]  # (N, V, ntri, 3)
        pt1 = vert[:, :, bfm.model['tri'][:, 1], :]  # (N, V, ntri, 3)
        pt2 = vert[:, :, bfm.model['tri'][:, 0], :]  # (N, V, ntri, 3)
        tri_normal = torch.cross(pt0 - pt1, pt0 - pt2, dim=-1)  # (N, V, ntri, 3). normal of each triangle
        tri_normal = torch.cat([tri_normal, torch.zeros_like(tri_normal[:, :, :1, :])], dim=2)  # (N, V, ntri + 1, 3)
        vert_tri_normal = tri_normal[:, :, bfm.tri_idx.ravel(), :].view(N, V, nver, bfm.tri_idx.shape[1], 3)
        normal = torch.sum(vert_tri_normal, dim=3)  # (N, V, nver, 3)

        loss = torch.cosine_similarity(normal[:, :, face_region_idx, :], normal_bfm_gt[:, :, face_region_idx, :], dim=3)
        loss = torch.mean(1. - loss, dim=2)  # (N, V)
        return loss

    def multi_normal_loss(self, opt_verts, normal_bfm_gt, vert_bfm_gt, bfm):
        reg_err = self.normal_loss(opt_verts[0].detach(), normal_bfm_gt, vert_bfm_gt, bfm)
        land_opt_err = self.normal_loss(opt_verts[1][-1].detach(), normal_bfm_gt, vert_bfm_gt, bfm)
        loss = 0.
        for opt_stage in range(1, len(opt_verts)):
            for opt_itr in range(len(opt_verts[opt_stage])):
                cur_loss = self.normal_loss(opt_verts[opt_stage][opt_itr], normal_bfm_gt, vert_bfm_gt, bfm)
                loss += torch.mean(cur_loss)
        feat_opt_err = cur_loss
        return loss, reg_err.detach(), land_opt_err.detach(), feat_opt_err.detach()

    def edge_loss(self, vert, vert_bfm_gt, bfm):
        N, V, nver, _ = vert.shape
        max_neib = bfm.neib_vert_idx.shape[1]

        # Compute neib_mask
        face_region_mask = bfm.face_region_mask.ravel().copy()                              # (nver,)
        face_region_mask = np.concatenate((face_region_mask, np.array([False])), axis=0)    # (nver + 1,)
        neib_mask = face_region_mask[bfm.neib_vert_idx.ravel()].reshape((nver, max_neib))
        neib_mask = torch.from_numpy(neib_mask).view(1, nver, max_neib).byte() \
                    [:, bfm.face_region_mask.ravel(), :].to(vert.device)

        # Compute edge distance of vert
        vert = vert.view(N * V, nver, 3)
        vert_t = torch.cat([vert, torch.zeros_like(vert[:, :1, :])], dim=1)             # (N * V, nver + 1, 3)
        vert_neib = vert_t[:, bfm.neib_vert_idx.ravel(), :].view(N * V, nver, max_neib, 3)
        vert_edge = torch.norm(vert.view(N * V, nver, 1, 3) - vert_neib, dim=3) \
                    [:, bfm.face_region_mask.ravel(), :]                                # (N * V, nver_face, max_neib)

        # Compute edge distance of vert_bfm_gt
        vert_bfm_gt = vert_bfm_gt.view(N * V, nver, 3)
        vert_t = torch.cat([vert_bfm_gt, torch.zeros_like(vert_bfm_gt[:, :1, :])], dim=1)   # (N * V, nver + 1, 3)
        vert_neib_bfm_gt = vert_t[:, bfm.neib_vert_idx.ravel(), :].view(N * V, nver, max_neib, 3)
        vert_edge_bfm_gt = torch.norm(vert_bfm_gt.view(N * V, nver, 1, 3) - vert_neib_bfm_gt, dim=3) \
                           [:, bfm.face_region_mask.ravel(), :]                         # (N * V, nver_face, max_neib)

        # Compute masked loss
        vert_edge = torch.where(torch.eq(neib_mask, 0), vert_edge_bfm_gt, vert_edge)
        eps = 1e-5
        loss = torch.mean(torch.abs(vert_edge / (vert_edge_bfm_gt + eps) - 1.))

        return loss

    def multi_edge_loss(self, opt_verts, vert_bfm_gt, bfm):
        loss = 0.
        for opt_stage in range(1, len(opt_verts)):
            for opt_itr in range(len(opt_verts[opt_stage])):
                loss += self.edge_loss(opt_verts[opt_stage][opt_itr], vert_bfm_gt, bfm)
        return loss


if __name__ == '__main__':
    # Pseudo code examples
    # Initialize loss function object
    criterion = LossFunction(mean_loss=True)

    # Inputs of losses -------------------------------------------------------------------------------------------------
    # Model outputs
    # list structures: [
    #                       mean face,
    #                       [level 1 itr 1,     level 1 itr 2,      level 1 itr 3],
    #                       [level 2 itr 1,     level 2 itr 2,      level 2 itr 3],
    #                       [level 3 itr 1,     level 3 itr 2,      level 3 itr 3]
    #                  ]
    opt_verts = < estimated mesh vertices >
    opt_verts_img = < estimated mesh vertices in image space >
    opt_full_face_vis_masks = < visibility masks of all vertices >
    opt_vis_masks = < visibility masks of vertices inside face region >

    # Ground truths
    vert_bfm_gt = < groundtruth mesh vertices in BFM topology >
    normal_bfm_gt = < groundtruth normals of vertices in BFM topology >
    kpts_gt = < detected 2D face landmarks >
    kpts_3d_gt = < detected 3D face landmarks >

    # Model components
    bfm = < model.opt_layer.bfm >
    # ------------------------------------------------------------------------------------------------------------------

    # Optimizer
    if < train pose regression network >:
        model.regressor.train()
    params = model.regressor.parameters() if < train pose regression network > else model.opt_layer.parameters()
    optimizer = torch.optim.Adam(params, lr=< learning rate = 2.0e-5 >)

    # Model forward
    N = < batch size = 2 >
    _, _, _, _, _, _, _, _, opt_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, _, _, _, _, _ = \
        model.forward(img, ori_img, kpts_gt, None, True if < train pose regression network > else False)

    # Compute losses
    vert_loss, _, _, _ = criterion.vert_loss(opt_verts, vert_bfm_gt, bfm.face_region_mask, align_mode='depth')
    align_vert_loss, _, _, _ = criterion.vert_loss(opt_verts, vert_bfm_gt, bfm.face_region_mask, align_mode='dense')
    land_loss = criterion.multi_land3d_loss(opt_verts_img, kpts_gt, kpts_3d_gt, bfm, opt_full_face_vis_masks, opt_vis_masks)
    normal_loss, normal_reg_err, normal_land_opt_err, normal_feat_opt_err = criterion.multi_normal_loss(opt_verts, normal_bfm_gt, vert_bfm_gt, bfm)
    edge_loss = criterion.multi_edge_loss(opt_verts, vert_bfm_gt, bfm)

    loss = vert_loss + align_vert_loss + 0.1 * land_loss + 100. * normal_loss + 1e-2 * edge_loss

    # Loss for pretraining pose regression network
    loss = criterion.land3d_loss(opt_verts_img[0], kpts_gt, kpts_3d_gt, bfm, opt_full_face_vis_masks[0], opt_vis_masks[0])
