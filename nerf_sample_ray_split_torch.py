from json import decoder
import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
from py360convert import e2c

########################################################################################################################
# ray batch sampling
########################################################################################################################
def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o  = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth

# sampling for training 6 images of cubemap in one pass
def get_rays_single_image_cubemap(H, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(H), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5 - H/2   # add half pixel, adjust center
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5 - H/2
    d = np.ones_like(u)
    # pixels = np.stack((u, v, d), axis=0)  # (3, H*W)
    rays_d_list = []
    # append rays_d in order of F, R, B, L, U, D
    # each (3, H*W)
    F = np.stack((u, v, d), axis=0).reshape((3, H, H))
    R = np.stack((d, v, -u), axis=0).reshape((3, H, H))
    B = np.stack((-u, v, -d), axis=0).reshape((3, H, H))
    L = np.stack((-d, v, u), axis=0).reshape((3, H, H))
    U = np.stack((u, -d, v), axis=0).reshape((3, H, H))
    D = np.stack((u, d, -v), axis=0).reshape((3, H, H))
    
    for i in range(H):
        d = np.concatenate((F[:, i], R[:, i], B[:, i], L[:, i], U[:, i], D[:, i]), axis=-1)
        # print(d)
        rays_d_list.append(d)
    rays_d = np.concatenate(rays_d_list, axis=-1) # (3, H*W*6)
    # rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), rays_d)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W*6)
    rays_d = rays_d.transpose((1, 0))  # (H*W*6, 3)
    rays_d_list.append(rays_d)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o  = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W*6, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W*6,)

    return rays_o, rays_d, depth

# sampling for training 6 images of cubemap each one at a time
def get_rays_single_image_cube(H, W, intrinsics, c2w, imgpath):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5 # add half pixel, adjust center
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    # assert(W==H)
    d = np.ones_like(u) * W / 2
    # pixels = np.stack((u, v, d), axis=0)  # (3, H*W)
    # append rays_d in order of F, R, B, L, U, D
    # each (3, H*W)
    # print(imgpath.split('_')[-1])
    if 'F' in imgpath.split('_')[-1]:
        rays_d = np.stack((u, v, d), axis=0) #Front
    elif 'R' in imgpath.split('_')[-1]:
        rays_d = np.stack((d, v, -u), axis=0) #Right
    elif 'B' in imgpath.split('_')[-1]:
        rays_d = np.stack((-u, v, -d), axis=0) #Back
    elif 'L' in imgpath.split('_')[-1]:
        rays_d = np.stack((-d, v, u), axis=0) #Left
    elif 'U' in imgpath.split('_')[-1]:
        rays_d = np.stack((u, -d, v), axis=0) #Up
    elif 'D' in imgpath.split('_')[-1]:
        rays_d = np.stack((u, d, -v), axis=0) #Down
    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), rays_d)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)
    

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o  = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth


# sampling for training 360 image mapped to sphere
def get_rays_single_image_360(H, W, intrinsics, c2w):
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5 - W/2   # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5 - H/2
    # set image center as (0, 0, 1) in camera world 
    
    u = (u / W) * 2 * np.pi # -pi ~ pi
    v = (v / H) * np.pi # -pi/2 ~ pi/2
    x = np.cos(-v) * np.sin(u)
    y = np.sin(-v)
    z = np.cos(-v) * np.cos(u)

    x_prime, y_prime, z_prime = x, y, z
    pixels = np.stack((x_prime, y_prime, z_prime), axis=0)
    pixels_t = torch.from_numpy(pixels).cuda()
    # pixels_t = torch.from_numpy(pixels)

    # print(c2w.get_device())
    # print(pixels_t.get_device())
    rays_d_t = torch.matmul(c2w.clone()[:3, :3], pixels_t)
    # print('sampling grad', rays_d_t.requires_grad, pixels_t.requires_grad, c2w.requires_grad)
    rays_d_t = torch.transpose(rays_d_t, 1, 0)

    
    rays_o_t = torch.reshape(c2w.clone()[:3, 3], (1,3))
    rays_o_t = torch.tile(rays_o_t, (rays_d_t.shape[0], 1))

    depth_t = torch.linalg.inv(c2w.clone())[2, 3]
    depth_t = depth_t * torch.ones((rays_o_t.shape[0],),dtype=torch.float32).cuda()

    # rays_o_t = rays_o_t.cuda()
    # rays_d_t = rays_d_t.cuda()
    # depth_t = depth_t.cuda()
    del pixels_t

    return rays_o_t, rays_d_t, depth_t
    # return pixel

class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics, c2w,
                       img_path=None,
                       resolution_level=1,
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None,
                       val=False):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        try:
            self.c2w_mat = torch.from_numpy(c2w).cuda()
        except:
            self.c2w_mat = c2w

        self.img_path = img_path
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth

        self.resolution_level = -1
        # if val: self.set_resolution_level(resolution_level)
        self.set_resolution_level(resolution_level)
    
    # def update_pose(self, pose):
    #     self.c2w_mat = pose
    #     self.set_resolution_level(1)

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:
                
                # change ground truth image here
                self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                self.mask = self.mask.reshape((-1))
            else:
                self.mask = None

            if self.min_depth_path is not None:
                self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
                self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            # changing samplin function here
            # self.rays_o, self.rays_d, self.depth = get_rays_single_image(self.H, self.W,
            #                                                              self.intrinsics, self.c2w_mat)
            # print('before sampling grad', self.c2w_mat.requires_grad)
            self.rays_o, self.rays_d, self.depth = get_rays_single_image_360(self.H, self.W,
                                                                         self.intrinsics, self.c2w_mat)
            # self.rays_o, self.rays_d, self.depth = get_rays_single_image_cubemap(self.H,
            #                                                              self.intrinsics, self.c2w_mat)
            # self.rays_o, self.rays_d, self.depth = get_rays_single_image_cube(self.H, self.W, self.intrinsics, 
            #                                                                   self.c2w_mat, self.img_path)
                                                                        

    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            return None
        
    def get_img_orig(self):
        if self.img_orig is not None:
            return self.img_orig.reshape((self.H_orig, self.W_orig, 3))
        else:
            return None

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            # min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])
            min_depth = 1e-4 * torch.ones_like(self.rays_d[..., 0]).cuda()

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                try:
                    ret[k] = torch.from_numpy(ret[k])
                except:
                    pass
        return ret

    def random_sample(self, N_rand, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            temp = self.H * self.W
            select_inds = np.random.choice(temp, size=(N_rand,), replace=False)

        # print('random sample grad', self.rays_o.requires_grad, self.c2w_mat.requires_grad)
        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]

        if self.img is not None:
            # print(self.img.shape, select_inds.shape)
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * torch.ones_like(rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('rgb', rgb),
            ('mask', mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path)
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret


# b = get_rays_single_image_cubemap(360, torch.randn((4,4)), torch.randn((4,4)))
# print(b[0].shape, b[1].shape, b[2].shape)