import numpy as np
import os
from math import *

def get_rotation_mat(deg_x,deg_y,deg_z):
    a = deg_x * pi / 180
    b = deg_y * pi / 180
    c = deg_z * pi / 180

    # giving noise
    a = a + np.random.normal(0, 0.01)
    b = b + np.random.normal(0, 0.01)
    c = c + np.random.normal(0, 0.01)

    mat = np.zeros((3,3))
    mat[0,0] = cos(b)*cos(c)
    mat[0,1] = sin(a)*sin(b)*cos(c) - cos(a)*sin(c)
    mat[0,2] = cos(a)*sin(b)*cos(c) + sin(a)*sin(c)
    mat[1,0] = cos(b)*sin(c)
    mat[1,1] = sin(a)*sin(b)*sin(c) + cos(a)*cos(c)
    mat[1,2] = cos(a)*sin(b)*sin(c) - sin(a)*cos(c)
    mat[2,0] = -sin(b)
    mat[2,1] = sin(a)*cos(b)
    mat[2,2] = cos(a)*cos(b)
    return mat

def get_tf_cams(pose_list, target_radius=1.):
    cam_centers = []
    for W2C in pose_list:
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale

def transform_pose(W2C, translate, scale):
    C2W = np.linalg.inv(W2C)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return np.linalg.inv(C2W)

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

pose_list = []

for f in sorted(os.listdir('data/dataset')):
    try:
        x,y,z = f.split(',')
        x = float(x)
        y = float(y)
        z = float(z)

        # giving noise
        x = x + np.random.normal(0, 0.01)
        y = y + np.random.normal(0, 0.01)
        z = z + np.random.normal(0, 0.01)
        
        pose = np.zeros((4,4))
        pose[:3,:3] = get_rotation_mat(87.8, 0, 176)
        pose[0,3] = x
        pose[1,3] = y
        pose[2,3] = z
        pose[3,3] = 1
        pose = np.linalg.inv(pose) # assuming c2w so convert to w2c

        pose_list.append(pose)
    except:
        pass

translate, scale = get_tf_cams(pose_list)

exp_dir = 'room6_noise'
base_dir_test = os.path.join('data/360', exp_dir, 'test', 'pose')
base_dir_train = os.path.join('data/360', exp_dir, 'train', 'pose')
if not os.path.exists(base_dir_test):
    os.makedirs(base_dir_test)
if not os.path.exists(base_dir_train):
    os.makedirs(base_dir_train)


for i in range(len(pose_list)):
    W2C = pose_list[i]
    W2C = transform_pose(W2C, translate, scale)
    C2W = np.linalg.inv(W2C)
    C2W = convert_pose(C2W)
    C2W = C2W.reshape(16)
    C2W_list = C2W.tolist()
    if i%20 == 0:
        pose_dir = os.path.join(base_dir_test, str(i+1).zfill(3) + '.txt')
    else:
        pose_dir = os.path.join(base_dir_train, str(i+1).zfill(3) + '.txt')

    pose_file = open(pose_dir, "w")
    for j in C2W_list:
        pose_file.write(str(j))
        pose_file.write(' ')
    pose_file.close()