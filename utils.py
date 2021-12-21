import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np


HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision
 

# misc utils
def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)


def normalize(x):
    min = x.min()
    max = x.max()

    return (x - min) / ((max - min) + TINY_NUMBER)


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
# gray2rgb = lambda x: np.tile(x[:,:,np.newaxis], (1, 1, 3))
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


########################################################################################################################
#
########################################################################################################################
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None):
    fig = Figure(figsize=(1.2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = ['{:3.2f}'.format(x) for x in tick_loc]
    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, append_cbar=False):
    if mask is not None:
        # vmin, vmax = np.percentile(x[mask], (1, 99))
        vmin = np.min(x[mask])
        vmax = np.max(x[mask])
        vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        x = np.clip(x, vmin, vmax)
        # print(vmin, vmax)
    else:
        vmin = x.min()
        vmax = x.max() + TINY_NUMBER

    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.zeros_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new, cbar


# tensor
def colorize(x, cmap_name='jet', append_cbar=False, mask=None):
    x = x.numpy()
    if mask is not None:
        mask = mask.numpy().astype(dtype=np.bool)
    x, cbar = colorize_np(x, cmap_name, mask)

    if append_cbar:
        x = np.concatenate((x, np.zeros_like(x[:, :5, :]), cbar), axis=1)

    x = torch.from_numpy(x)
    return x


from py360convert import e2c, c2e
from PIL import Image
from os import listdir, remove, path, makedirs
import shutil
arr = ('F', 'R', 'B', 'L', 'U', 'D')
def save_cubemap(dir, w):
    dirs = listdir(dir)
    print(dirs)
    for img_dir in dirs:
        img_dir = dir + '/' + img_dir
        print(img_dir)
        img = np.array(Image.open(img_dir))
        img_cubelist = e2c(img, w, mode='bilinear', cube_format='list')
        for i in range(1):
            img_pil = Image.fromarray(img_cubelist[i].astype('uint8'), 'RGB')
            img_new_dir = img_dir.split('.')[0] + '_' + arr[i] + '.jpg'
            print(img_new_dir)
            img_pil.save(img_new_dir)
        remove(img_dir)
        

def dup_file(root):
    dirs = listdir(root)
    print(dirs)
    for dir in dirs:
        dir = root + '/' + dir
        for i in range(1):
            new_dir = dir.split('.')[0] + '_' + arr[i] + '.' + dir.split('.')[-1]
            print(new_dir)
            shutil.copyfile(dir, new_dir)
        remove(dir)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def preproces_cube(dir):
    dirs = listdir(dir)
    # print(dirs)
    for img_dir in dirs:
        if 'bg' in img_dir or 'fg' in img_dir: continue
        img_full_dir = dir + '/' + img_dir
        if not path.isfile(img_full_dir): continue
        # print(img_dir.split('_'))
        img_id, img_pos = img_dir.split('_')[0], img_dir.split('_')[1].split('.')[0]
        # print(img_id, img_pos)
        img_folder_dir = dir + '/' + img_id
        if not path.exists(img_folder_dir):
            makedirs(img_folder_dir)
        
        img_new_dir = img_folder_dir + '/' + img_pos + '.jpg'
        # print(img_new_dir)
        img = Image.open(img_full_dir)
        img.save(img_new_dir)

        


def cube2equi(dir, cube_format='list'):
    pass   
    dirs = listdir(dir)
    # print(dirs)
    img_dict = dict()
    for img_dir in dirs:
        # print(img_dir)
        # if 'bg' in img_dir or 'fg' in img_dir: continue
        img_full_dir = dir + '/' + img_dir

        # if not path.isfile(img_full_dir): continue
        # # print(img_dir)
        # img_id, img_pos = img_dir.split('_')[0], img_dir.split('_')[1].split('.')[0]
        # print(img_id, img_pos)
        # print(img_dir)
        img = np.array(Image.open(img_full_dir))
        img_pos = img_dir.split('.')[0]
        if img_pos not in arr: continue
        # print(img_pos)
        img_dict[img_pos] = img
        # print(img.shape)
    # print(img_dict.keys())
    img_list = []
    for i in range(len(img_dict)):
        img_list.append(img_dict[arr[i]])
    # print(dir, len(img_list))
    img_equi = c2e(img_list, 720, 1440, cube_format=cube_format)
    img_equi_pil = Image.fromarray(img_equi.astype('uint8'), 'RGB')
    img_name = dir.split('/')[-1]
    # print(img_name)
    img_new_dir = dir + '/' + img_name + '.jpg'
    img_equi_pil.save(img_new_dir)
    return img_new_dir, img_name

def cube2equi_horizon(dir, cube_format='horizon'):
    pass   
    dirs = listdir(dir)
    # print(dirs)
    img_dict = dict()
    new_folder = dir + '/panorama'
    if not path.exists(new_folder): makedirs(new_folder)
    for img_dir in dirs:
        # print(img_dir)
        if 'bg' in img_dir or 'fg' in img_dir: continue
        img_full_dir = dir + '/' + img_dir
        # img_new_dir = dir + '/panorama' + img_dir
        img_new_dir = new_folder + '/' + img_dir

        # if not path.isfile(img_full_dir): continue
        # # print(img_dir)
        # img_id, img_pos = img_dir.split('_')[0], img_dir.split('_')[1].split('.')[0]
        # print(img_id, img_pos)
        # print(img_dir)
        img = np.array(Image.open(img_full_dir))
        # print(img_pos)
        # print(img.shape)
        img_equi = c2e(img, 720, 1440, cube_format=cube_format)
    # print(img_dict.keys())
    
    
        img_equi_pil = Image.fromarray(img_equi.astype('uint8'), 'RGB')
        img_equi_pil.save(img_new_dir)
        
    
    return


def full_cube2equi(dir, cube_format='list'):
    dirs = listdir(dir)
    print(dirs)
    equi_list = []
    new_folder = dir + '/panorama'
    if not path.exists(new_folder): makedirs(new_folder)
    for folder in dirs:
        full_dir = dir + '/' + folder
        if path.isfile(full_dir): continue

        # print(full_dir)
        if 'panorama' in folder: continue
        img_dir, img_name = cube2equi(full_dir, cube_format=cube_format)
        new_file = new_folder + '/' + img_name + '.jpg'
        # print(new_file)
        shutil.copyfile(img_dir, new_file)


def full_cube2equi_horizon(dir, cube_format='horizon'):
    dirs = listdir(dir)
    print(dirs)
    new_folder = dir + '/panorama'
    if not path.exists(new_folder): makedirs(new_folder)
    for img_dir in dirs:
        img_full_dir = dir + '/' + img_dir
        if 'panorama' in img_dir: continue
        if not path.isfile(img_full_dir): continue
        if 'bg' in img_dir or 'fg' in img_dir or 'depth' in img_dir: continue

        img = np.array(Image.open(img_full_dir))
        new_img = c2e(img, 720, 1440, cube_format=cube_format)
        new_img = Image.fromarray(new_img.astype('uint8'), 'RGB')
        new_file = new_folder + '/' + img_dir
        # print(new_file)
        new_img.save(new_file)

def get_full_psnr(test_dir, gt_dir):
    test_imgs = listdir(test_dir)
    gt_imgs = listdir(gt_dir)
    test_imgs.sort()
    gt_imgs.sort()
    assert len(test_imgs) == len(gt_imgs)
    total_psnr = []
    for i in range(len(test_imgs)):
        test_img_dir = test_dir + '/' + test_imgs[i]
        gt_img_dir = gt_dir + '/' + gt_imgs[i]
        # print(test_imgs[i], gt_imgs[i])
        test_img, gt_img = np.array(Image.open(test_img_dir)) / 255., np.array(Image.open(gt_img_dir)) / 255.
        # print(test_img.shape)
        # print(np.mean((test_img - gt_img) * (test_img - gt_img)))

        # psnr = mse2psnr(np.mean((test_img - gt_img) * (test_img - gt_img)))
        psnr = mse2psnr(np.mean((test_img - gt_img) * (test_img - gt_img)))
        print('PSNR of {}: {}'.format(test_imgs[i], psnr))
        total_psnr.append(psnr)
    avg = sum(total_psnr) / len(total_psnr)
    print('Average PSNR: ', avg)
        

    

# save_cubemap('data/360/room4_downscale_cube/train/rgb', 400)
# dup_file('data/360/room4_downscale_cube/test/intrinsics')
# cube2equi('logs/room4_downscale_cylinder_cube/render_test_500000/03')
# preproces_cube('logs/room4_downscale_cylinder_cube/render_train_500000_train')
# full_cube2equi('logs/room4_downscale_cylinder_cube/render_train_500000_train', cube_format='list')
# cube2equi_horizon('logs/room4_downscale_sphere_cubemap_2/render_train_440000')
# get_full_psnr('logs/africa/render_test_450000/image/', 'data/lf_data/africa/test/rgb')
# get_full_psnr('logs/africa_cylinder_7/render_test_355000/panorama', 'data/lf_data/africa/test/rgb')

