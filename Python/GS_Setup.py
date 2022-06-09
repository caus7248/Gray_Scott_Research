# -*- coding: utf-8 -*-
"""


@author: mcausley
"""

import numpy as np
from PIL import Image

__all__ = [
    "intial_condition",
    "jiggle",
    "update",
    "laplacian",
    "Im_Plot",
]

def intial_condition(num_points):
    u =np.ones((num_points,num_points))
    v =np.zeros((num_points,num_points))
    bot = int(num_points*0.65)
    top = int(num_points*0.75)
    lft = int(num_points*0.65)
    rgt = int(num_points*0.75)
    u[bot:top, lft:rgt] += 0.1
    v[bot:top, lft:rgt] += 0.5
    bot = int(num_points*0.25)
    top = int(num_points*0.35)
    lft = int(num_points*0.25)
    rgt = int(num_points*0.35)
    u[bot:top, lft:rgt] += 0.1
    v[bot:top, lft:rgt] += 0.5
    return u, v

def jiggle(v,num_points):
    bot = int(num_points*0.4)
    top = int(num_points*0.5)
    lft = int(num_points*0.45)
    rgt = int(num_points*0.55)
    v[bot:top, lft:rgt] += 0.1
    return v

def update(u, v, time_step = 0.5, D_u = 0.2, D_v = 0.1, k = 0.045, F = 0.014):
    du, dv = ( D_u*laplacian(u)-u*v*v+(1-u)*F,
               D_v*laplacian(v)+u*v*v-(F+k)*v
               )
    return u+time_step*du, v+time_step*dv

def laplacian(u):
    """
    Given a discrete rectangular field u, compute the discrete Laplacian
    based on a 9-point finite difference stencil
    ----------
    u : the field (2D array)

    Returns
    -------
    du:  the Laplacian of u (2D array)
    """
    u_cross =(
        np.roll(u, ( 1, 0), axis=(0, 1)) +
        np.roll(u, ( 0, 1), axis=(0, 1)) +
        np.roll(u, (-1, 0), axis=(0, 1)) +
        np.roll(u, ( 0,-1), axis=(0, 1)) - 
        4*u
        )
    u_diag =(
        np.roll(u, ( 1, 1), axis=(0, 1)) +
        np.roll(u, ( 1,-1), axis=(0, 1)) +
        np.roll(u, (-1, 1), axis=(0, 1)) +
        np.roll(u, (-1,-1), axis=(0, 1)) - 
        4*u
        )
    return ( (4*u_cross+ u_diag)/6 )
    # return  u_cross


def Im_Plot(u):
    im = Image.fromarray(u*255)
    return  im

def get_Fk(exp_no):
    Fk =[[0.014, 0.045, 'waves'],
         [0.11, 0.0523, 'solitons'],
         [0.09, 0.059, 'soap bubbles']
         ]
    return Fk[:][exp_no-1]