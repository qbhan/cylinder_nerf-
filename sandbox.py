# from utils import save_front

# save_front('data/360/room6_front/train/rgb', 400)


import torch
import numpy as np
from utils import make_c2w

device = torch.device('cpu')
model = torch.load("logs/room6_colmap_posetrain_1/pose90000.pth", map_location = device)

init_pose = model['module.init_c2w']
R = model['module.r']
t = model['module.t']

print(R)

savdir = 'sandbox_data/'
for i in range(init_pose.shape[0]):
    pose = make_c2w(R[i], t[i])
    pose = pose @ init_pose[i]
    pose = pose.numpy()
    pose = np.reshape(pose, 16)
    savdir = 'sandbox_data/' + str(i+1).zfill(3) + '.txt'

    f = open(savdir, "w")
    for j in pose:
        f.write(str(j.item()))
        f.write(" ")

# print(trained_pose)
# b = a['module.r'].tolist()
# # a = model.load_state_dict("logs/room6_noise_posetrain/pose170000.pth")
# f = open('temp.txt', "w")
# for i in b:
#     # print(i)
#     f.write(str(i))