# from utils import save_front
# save_front('data/360/room6_front/train/rgb', 400)


import os
import numpy as np
from utils import save_front
from py360convert import e2c, c2e, e2p
from PIL import Image


def get_info_from_pose(exp_dir):
    target_dir = os.path.join("data/360", exp_dir, "test/pose")

    res = open(os.path.join("data/360", exp_dir, "test", "res.txt"), "w")

    for f in sorted(os.listdir(target_dir)):
        pose_file = open(os.path.join(target_dir, f), "r")
        pose_str = pose_file.readline()
        pose_list = []
        for i in pose_str.split(" "):
            try:
                pose_list.append(float(i))
            except:
                pass
        
        if len(pose_list) == 16:
            pose = np.array(pose_list)
            pose = pose.reshape(4,4)
            t = pose[0:3, 3]
            t = t / np.linalg.norm(t)

            axis = np.array([pose[2,1] - pose[1,2], pose[0,2] - pose[2,0], pose[1,0] - pose[0,1]])
            axis = axis / np.linalg.norm(axis)

            angle = np.arccos((pose[0,0] + pose[1,1] + pose[2,2] - 1) / 2)
            temp = "t: %f, %f, %f, axis: %f, %f, %f, angle: %f\n"%(t[0], t[1], t[2], axis[0], axis[1], axis[2], angle)
            res.write(temp)

def get_2D_from_360(dir):
    dirs = os.listdir(dir)
    for img_dir in dirs:
        img_dir = dir + '/' + img_dir
        print(img_dir)
        img = np.array(Image.open(img_dir))
        img_planer = e2p(img, (120, 120), 0, 0, (500, 500), 0, "bilinear")
        img_pil = Image.fromarray(img_planer.astype('uint8'), 'RGB')
        img_new_dir = img_dir.split('.')[0] + '_p.jpg'
        print(img_new_dir)
        img_pil.save(img_new_dir)
        os.remove(img_dir)

# get_2D_from_360('data/360/room6_copy/train/rgb')
get_info_from_pose("room6_colmap2")