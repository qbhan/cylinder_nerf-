from cv2 import DIST_MASK_3
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from utils import TINY_NUMBER, HUGE_NUMBER, make_c2w
from collections import OrderedDict
from nerf_network import Embedder, MLPNet
import os
import logging
logger = logging.getLogger(__package__)


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    # print(depth[0,:])
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]
    # print(rot_axis.shape, p_sphere.shape)
    
    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


def depth2pts_outside_cylinder(ray_o, ray_d, inv_r):
    
    # projection to x, y plane of cylinder
    ray_d_2d = ray_d[:, :, :2]
    ray_o_2d = ray_o[:, :, :2]
    # ratio = torch.norm(ray_d, dim=-1) / torch.norm(ray_d_2d, dim=-1) # ratio between 3d and 2d projection

    # do same as above, but on projected 2d
    # note: d1 becomes negative if this mid point is behind camera

    d1 = -torch.sum(ray_d_2d * ray_o_2d, dim=-1) / torch.sum(ray_d_2d * ray_d_2d, dim=-1)
    p_mid_2d = ray_o_2d + d1.unsqueeze(-1) * ray_d_2d
    p_mid_3d = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_2d_norm = torch.norm(p_mid_2d, dim=-1)
    p_mid_3d_norm = torch.norm(p_mid_3d, dim=-1)
    ray_d_2d_cos = 1. / torch.norm(ray_d_2d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_2d_norm * p_mid_2d_norm) * ray_d_2d_cos
    p_sphere_3d = ray_o + (d1 + d2).unsqueeze(-1) * ray_d
    
    r = 1 / (TINY_NUMBER + inv_r)
    r = r.unsqueeze(-1)
    # print(r.shape)
    d3 = torch.sqrt(torch.square(torch.sum(ray_d_2d * ray_o_2d, dim=-1)) + torch.sum(ray_d_2d * ray_d_2d, dim=-1) * (torch.sum(r * r, dim=-1) - torch.sum(ray_o_2d * ray_o_2d, dim=-1))) / (torch.sum(ray_d_2d * ray_d_2d, dim=-1) + TINY_NUMBER)
    # t = d1 + d3
    # print(ray_o.isnan().any(), ray_d.isnan().any(), inv_r.isnan().any(), d3.isnan().any())
    p_cylinder = ray_o + (d1 + d3).unsqueeze(-1) * ray_d
    p_cylinder_2d_norm = torch.norm(p_cylinder[:, :, :2], dim=-1)
    # print(p_cylinder.shape, p_cylinder_2d_norm.shape)
    p_cylinder = p_cylinder / p_cylinder_2d_norm.unsqueeze(-1)
    pts = torch.cat((p_cylinder, inv_r.unsqueeze(-1)), dim=-1)

    depth_real = torch.norm(ray_o + (d1 + d3).unsqueeze(-1) * ray_d, dim=-1)
    # print(depth_real)
    
    return pts, depth_real


class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, bg_r_vals=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        is_nan = ray_o.isnan().any() and ray_d.isnan().any() and fg_z_max.isnan().any() and fg_z_vals.isnan().any() and bg_z_vals.isnan().any()
        # print('input', is_nan)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm      # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        # print(fg_pts)
        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        fg_raw = self.fg_net(input)
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]), dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)   # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T     # [..., N_samples]
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]

        # render background
        # change
        # 1/r to cylinderical coordinate.
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        # bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        bg_pts, _ = depth2pts_outside_cylinder(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        # print('cylinder background input', bg_ray_o.isnan().any() and bg_ray_d.isnan().any() and bg_z_vals.isnan().any())
        # print('cylinder background', bg_pts.isnan().any())
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2,])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1,])           # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input)
        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # print(bg_raw)
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        bg_depth_map = bg_lambda * bg_depth_map
        rgb_map = fg_rgb_map + bg_rgb_map

        ret = OrderedDict([('rgb', rgb_map),            # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),      # below are for logging
                           ('fg_depth', fg_depth_map),
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map),
                           ('bg_lambda', bg_lambda)])
        # isnan = rgb_map.isnan().any() and fg_weights.isnan().any() and bg_weights.isnan().any() and fg_rgb_map.isnan().any() and fg_depth_map.isnan().any() and bg_rgb_map.isnan().any() and bg_depth_map.isnan().any() and bg_lambda.isnan().any()
        # print(isnan)
        return ret


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNet(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert(img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5 # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w

class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad, fx_only, order=2, init_focal=None):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.

        if self.fx_only:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
        else:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                    coe_y = torch.tensor(np.sqrt(init_focal / float(H)), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                    coe_y = torch.tensor(init_focal / float(H), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )

    def forward(self, i=None):  # the i=None is just to enable multi-gpu training
        if self.fx_only:
            if self.order == 2:
                fxfy = torch.stack([self.fx ** 2 * self.W, self.fx ** 2 * self.W])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fx * self.W])
        else:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fy * self.H])
        return fxfy