# -*- coding: utf-8 -*-
"""


@author: mcausley
"""

import sys
import os
sys.path.append(os.getcwd())
import GS_Setup as GS
import matplotlib.pyplot as plt

# Modeling parameters: diffusion coefficients
D_u = 2e-1
D_v = 1e-1

# feed and kill rate vary with experiment number. See GS_Setup.py
exp_no = 3
F, k, folder_name = GS.get_Fk(exp_no)
print((F,k, folder_name))

# Numerical parameters:  time step, grid points, and time frames
time_step = 0.5 # time steps > 1.0 create numerical instability
num_points = 256
# num_frames = 99000
num_frames = 2

# Create a subfolder for storing images
path = os.getcwd() + '/Images/' + folder_name
if os.path.exists(path):
    print("The image directory is already created.")
else:
    os.makedirs(path)
    print("Created new Image directory.")


# Initialization
ic_no = 4
x_no = .5
y_no = .5
size_no = .1
u, v = GS.intial_condition(num_points, ic_no, x_no, y_no, size_no)

# Now evolve the initial state, and capture images
for i in range(1,num_frames):
    u,v = GS.update(u, v, time_step, D_u, D_v, k, F)
    if i == 400 or i== 800 or i== 1100:
        v = GS.jiggle(v,num_points)
        
    # if i % 100 == 0:         # capture an image every 100 time steps
    #     im = GS.Im_Plot(u)
    #     im_name = 'GS_' + folder_name + '_' + str(int(i/100)).zfill(3) + '.png'
    #     plt.imsave('Images/'+ folder_name + '/' + im_name, im, cmap='gray')
    im = GS.Im_Plot(u)
    im_name = 'GS_' + folder_name + '_' + str(int(i/100)).zfill(3) + '.png'
    plt.imsave('Images/'+ folder_name + '/' + im_name, im, cmap='gray')
