import os
import cv2
import numpy as np
from frame_seq_data import FrameSeqData
from seq_data.sun3d.read_util import read_sun3d_depth
import core_3dv.camera_operator as cam_opt
import core_3dv.camera_operator_gpu as cam_opt_gpu
from visualizer.visualizer_2d import show_multiple_img
from core_io.depth_io import load_depth_from_png

valid_set_dir = '/home/ziqianb/Desktop/datasets/tgz_target/'
valid_seq_name = 'rgbd_dataset_freiburg1_desk'

seq = FrameSeqData(os.path.join(valid_set_dir, valid_seq_name, 'seq.json'))

frame_a = seq.frames[5]
frame_b = seq.frames[20]

Tcw_a = seq.get_Tcw(frame_a)
Tcw_b = seq.get_Tcw(frame_b)
K = seq.get_K_mat(frame_a)

img_a = cv2.imread(os.path.join(valid_set_dir, seq.get_image_name(frame_a))).astype(np.float32) / 255.0
img_b = cv2.imread(os.path.join(valid_set_dir, seq.get_image_name(frame_b))).astype(np.float32) / 255.0
depth_a = load_depth_from_png(os.path.join(valid_set_dir, seq.get_depth_name(frame_a)), div_factor=5000.0)
depth_b = load_depth_from_png(os.path.join(valid_set_dir, seq.get_depth_name(frame_b)), div_factor=5000.0)

rel_T = cam_opt.relateive_pose(Tcw_a[:3, :3], Tcw_a[:3, 3], Tcw_b[:3, :3], Tcw_b[:3, 3])
wrap_b2a, _ = cam_opt.wrapping(img_a, img_b, depth_a, K, rel_T[:3, :3], rel_T[:3, 3])
dense_a2b, _ = cam_opt.dense_corres_a2b(depth_a, K, Tcw_a, Tcw_b)
overlap_marks = cam_opt.mark_out_bound_pixels(dense_a2b, depth_a)
overlap_marks = overlap_marks.astype(np.float32)
overlap_ratio = cam_opt.photometric_overlap(depth_a, K, Tcw_a, Tcw_b)
print(overlap_ratio)

# show_multiple_img([{'img': img_a, 'title': 'a'},
#                    {'img': img_b, 'title': 'b'},
#                    {'img': wrap_b2a, 'title':'a2b'},
#                    {'img': overlap_marks, 'title':'overlap', 'cmap':'gray'}], title='View', num_cols=4)

#
H, W, C = img_a.shape

""" Torch
"""
import torch
Tcw_a = torch.from_numpy(Tcw_a).cuda()
Tcw_b = torch.from_numpy(Tcw_b).cuda()
K = torch.from_numpy(K).cuda()
img_a = torch.from_numpy(img_a).cuda()
img_b = torch.from_numpy(img_b).cuda()
depth_a = torch.from_numpy(depth_a).cuda().view(H, W)
depth_b = torch.from_numpy(depth_b).cuda().view(H, W)

wrap_b2a, _ = cam_opt_gpu.wrapping(img_b, depth_a, K, Tcw_a, Tcw_b)
dense_a2b, _ = cam_opt_gpu.dense_corres_a2b(depth_a, K, Tcw_a, Tcw_b)
overlap_marks = cam_opt_gpu.mark_out_bound_pixels(dense_a2b, depth_a)
overlap_marks = overlap_marks.float()
overlap_ratio = cam_opt_gpu.photometric_overlap(depth_a, K, Tcw_a, Tcw_b)
print(overlap_ratio)

show_multiple_img([{'img': img_a.cpu().numpy(), 'title': 'a'},
                   {'img': img_b.cpu().numpy(), 'title': 'b'},
                   {'img': wrap_b2a.cpu().numpy(), 'title':'a2b'},
                   {'img': overlap_marks.cpu().numpy(), 'title':'overlap', 'cmap':'gray'}], title='View', num_cols=4)
