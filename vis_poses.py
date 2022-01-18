import torch
import numpy as np
import os
from data_loader_split_torch import find_files
from utils import make_c2w, SO3_to_quat, convert3x4_4x4

import open3d as o3d
from third_party.ATE.align_utils import alignTrajectory
from third_party.ATE.compute_trajectory_errors import compute_absolute_error
from third_party.ATE.results_writer import compute_statistics

def pts_dist_max(pts):
    """
    :param pts:  (N, 3) torch or np
    :return:     scalar
    """
    if torch.is_tensor(pts):
        dist = pts.unsqueeze(0) - pts.unsqueeze(1)  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = dist.norm(dim=1)  # (N, )
        max_dist = dist.max()
    else:
        dist = pts[None, :, :] - pts[:, None, :]  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = np.linalg.norm(dist, axis=1)  # (N, )
        max_dist = dist.max()
    return max_dist


def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method='sim3')

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    return traj_c_aligned  # (N1, 4, 4)


def align_scale_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    '''Scale c to b using the scale from a to b.
    :param traj_a:      (N0, 3/4, 4) torch tensor
    :param traj_b:      (N0, 3/4, 4) torch tensor
    :param traj_c:      None or (N1, 3/4, 4) torch tensor
    :return:
        scaled_traj_c   (N1, 4, 4)   torch tensor
        scale           scalar
    '''
    if traj_c is None:
        traj_c = traj_a.clone()

    t_a = traj_a[:, :3, 3]  # (N, 3)
    t_b = traj_b[:, :3, 3]  # (N, 3)

    # scale estimated poses to colmap scale
    # s_a2b: a*s ~ b
    scale_a2b = pts_dist_max(t_b) / pts_dist_max(t_a)

    traj_c[:, :3, 3] *= scale_a2b

    if traj_c.shape[1] == 3:
        traj_c = convert3x4_4x4(traj_c)  # (N, 4, 4)

    return traj_c, scale_a2b  # (N, 4, 4)

def compute_ate(c2ws_a, c2ws_b, align_a2b=None):
    """Compuate ate between a and b.
    :param c2ws_a: (N, 3/4, 4) torch
    :param c2ws_b: (N, 3/4, 4) torch
    :param align_a2b: None or 'sim3'. Set to None if a and b are pre-aligned.
    """
    if align_a2b == 'sim3':
        c2ws_a_aligned = align_ate_c2b_use_a2b(c2ws_a, c2ws_b)
        R_a_aligned = c2ws_a_aligned[:, :3, :3].cpu().numpy()
        t_a_aligned = c2ws_a_aligned[:, :3, 3].cpu().numpy()
    else:
        R_a_aligned = c2ws_a[:, :3, :3].cpu().numpy()
        t_a_aligned = c2ws_a[:, :3, 3].cpu().numpy()
    R_b = c2ws_b[:, :3, :3].cpu().numpy()
    t_b = c2ws_b[:, :3, 3].cpu().numpy()

    quat_a_aligned = SO3_to_quat(R_a_aligned)
    quat_b = SO3_to_quat(R_b)

    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = compute_absolute_error(t_a_aligned,quat_a_aligned,
                                                                              t_b, quat_b)
    stats_tran = compute_statistics(e_trans)
    stats_rot = compute_statistics(e_rot)
    stats_scale = compute_statistics(e_scale_perc)

    return stats_tran, stats_rot, stats_scale  # dicts


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def get_camera_frustum_opengl_coord(H, W, fx, fy, W2C, frustum_length=0.5, color=np.array([0., 1., 0.])):
    '''X right, Y up, Z backward to the observer.
    :param H, W:
    :param fx, fy:
    :param W2C:             (4, 4)  matrix
    :param frustum_length:  scalar: scale the frustum
    :param color:           (3,)    list, frustum line color
    :return:
        frustum_points:     (5, 3)  frustum points in world coordinate
        frustum_lines:      (8, 2)  8 lines connect 5 frustum points, specified in line start/end index.
        frustum_colors:     (8, 3)  colors for 8 lines.
    '''
    hfov = np.rad2deg(np.arctan(W / 2. / fx) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / fy) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum in camera space in homogenous coordinate (5, 4)
    frustum_points = np.array([[0., 0., 0., 1.0],                          # frustum origin
                               [-half_w, half_h,  -frustum_length, 1.0],   # top-left image corner
                               [half_w, half_h,   -frustum_length, 1.0],   # top-right image corner
                               [half_w, -half_h,  -frustum_length, 1.0],   # bottom-right image corner
                               [-half_w, -half_h, -frustum_length, 1.0]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])  # (8, 2)
    frustum_colors = np.tile(color.reshape((1, 3)), (frustum_lines.shape[0], 1))  # (8, 3)

    # transform view frustum from camera space to world space
    C2W = np.linalg.inv(W2C)
    frustum_points = np.matmul(C2W, frustum_points.T).T  # (5, 4)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]  # (5, 3)  remove homogenous coordinate
    return frustum_points, frustum_lines, frustum_colors


def draw_camera_frustum_geometry(c2ws, H, W, fx=600.0, fy=600.0, frustum_length=0.5,
                                 color=np.array([29.0, 53.0, 87.0])/255.0, draw_now=False, coord='opengl'):
    '''
    :param c2ws:            (N, 4, 4)  np.array
    :param H:               scalar
    :param W:               scalar
    :param fx:              scalar
    :param fy:              scalar
    :param frustum_length:  scalar
    :param color:           None or (N, 3) or (3, ) or (1, 3) or (3, 1) np array
    :param draw_now:        True/False call o3d vis now
    :return:
    '''
    N = c2ws.shape[0]

    num_ele = color.flatten().shape[0]
    if num_ele == 3:
        color = color.reshape(1, 3)
        color = np.tile(color, (N, 1))

    frustum_list = []
    if coord == 'opengl':
        for i in range(N):
            frustum_list.append(get_camera_frustum_opengl_coord(H, W, fx, fy,
                                                                W2C=np.linalg.inv(c2ws[i]),
                                                                frustum_length=frustum_length,
                                                                color=color[i]))
    else:
        print('Undefined coordinate system. Exit')
        exit()

    frustums_geometry = frustums2lineset(frustum_list)

    if draw_now:
        o3d.visualization.draw_geometries([frustums_geometry])

    return frustums_geometry  # this is an o3d geometry object.

def get_poses(basedir, scene, split):
    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    split_dir = '{}/{}/{}'.format(basedir, scene, split)
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
    cam_cnt = len(pose_files)
    pose_array = np.zeros((cam_cnt, 4, 4))
    for i in range(cam_cnt):
        pose_array[i,:,:] = parse_txt(pose_files[i])

    poses = torch.Tensor(pose_array)
    return cam_cnt, poses



datadir = "./data/360"
scene = "room6_colmap"
split = "train"
expdir = "room6_colmap_posetrain_3"
iteration = "pose430000.pth"

device = torch.device('cpu')
model = torch.load(os.path.join("logs", expdir, "pose", iteration), map_location = device)

c2ws_cmp = model['module.init_c2w']
R = model['module.r']
t = model['module.t']
N_imgs = c2ws_cmp.shape[0]
fxfy = [1000, 500]
H = 500
W = 1000

c2ws_est = torch.stack([make_c2w(R[i], t[i])@c2ws_cmp[i] for i in range(N_imgs)])

 # scale estimated poses to unit sphere
ts_est = c2ws_est[:, :3, 3]  # (N, 3)
c2ws_est[:, :3, 3] /= pts_dist_max(ts_est)
c2ws_est[:, :3, 3] *= 2.0

'''Define camera frustums'''
frustum_length = 0.1
est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255

'''Align est traj to colmap traj'''
c2ws_est_to_draw_align2cmp = c2ws_est.clone()
c2ws_est_aligned = align_ate_c2b_use_a2b(c2ws_est, c2ws_cmp)  # (N, 4, 4)
c2ws_est_to_draw_align2cmp = c2ws_est_aligned

# compute ate
stats_tran_est, stats_rot_est, _ = compute_ate(c2ws_est_aligned, c2ws_cmp, align_a2b=None)
print('From est to colmap: tran err {0:.3f}, rot err {1:.2f}'.format(stats_tran_est['mean'],
                                                                        stats_rot_est['mean']))

frustum_est_list = draw_camera_frustum_geometry(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W,
                                                fxfy[0], fxfy[1],
                                                frustum_length, est_traj_color)
frustum_colmap_list = draw_camera_frustum_geometry(c2ws_cmp.cpu().numpy(), H, W,
                                                    fxfy[0], fxfy[1],
                                                    frustum_length, cmp_traj_color)

geometry_to_draw = []
geometry_to_draw.append(frustum_est_list)
geometry_to_draw.append(frustum_colmap_list)

'''o3d for line drawing'''
t_est_list = c2ws_est_to_draw_align2cmp[:, :3, 3]
t_cmp_list = c2ws_cmp[:, :3, 3]

'''line set to note pose correspondence between two trajs'''
line_points = torch.cat([t_est_list, t_cmp_list], dim=0).cpu().numpy()  # (2N, 3)
line_ends = [[i, i+N_imgs] for i in range(N_imgs)]  # (N, 2) connect two end points.
# line_color = np.zeros((scene_train.N_imgs, 3), dtype=np.float32)
# line_color[:, 0] = 1.0

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector(line_ends)
# line_set.colors = o3d.utility.Vector3dVector(line_color)

geometry_to_draw.append(line_set)
o3d.visualization.draw_geometries(geometry_to_draw)