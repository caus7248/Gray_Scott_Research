# -*- coding: utf-8 -*-
"""


@author: mcausley
"""

from re import U
from matplotlib.pyplot import switch_backend
import numpy as np
from PIL import Image

__all__ = [
    "intial_condition",
    "jiggle",
    "update",
    "laplacian",
    "Im_Plot",
]

def intial_condition(num_points, ic_no, x_no, y_no, size_no, shape_no = 1, sharpness = 10):
    def one() : 
        u = np.ones((num_points, num_points))
        v = np.zeros((num_points, num_points))
        bot = int(num_points*(.5 + size_no * 1.5))
        top = int(num_points*(.5 + size_no * 2.5))
        lft = int(num_points*(.5 + size_no * 1.5))
        rgt = int(num_points*(.5 + size_no * 2.5))
        if(bot < 0) : 
            bot = 0
        if(top > 256) : 
            top = 256
        if(lft < 0) : 
            lft = 0
        if(rgt > 256) : 
            rgt = 256
        u[bot:top, lft:rgt] += 0.1
        v[bot:top, lft:rgt] += 0.5
        bot = int(num_points*(.5 - size_no * 2.5))
        top = int(num_points*(.5 - size_no * 1.5))
        lft = int(num_points*(.5 - size_no * 2.5))
        rgt = int(num_points*(.5 - size_no * 1.5))
        if(bot < 0) : 
            bot = 0
        if(top > 256) : 
            top = 256
        if(lft < 0) : 
            lft = 0
        if(rgt > 256) : 
            rgt = 256
        u[bot:top, lft:rgt] += 0.1
        v[bot:top, lft:rgt] += 0.5
        u = np.roll(u, (int((x_no - .5) * num_points), int((y_no - .5) * num_points)), axis=(0, 1))
        v = np.roll(v, (int((x_no - .5) * num_points), int((y_no - .5) * num_points)), axis=(0, 1))
        return u, v

    def two() : 
        u = np.ones((num_points, num_points))
        v = np.zeros((num_points, num_points))
        bot = int(num_points*(.5 - size_no * .5))
        top = int(num_points*(.5 + size_no * .5))
        lft = int(num_points*(.5 - size_no * .5))
        rgt = int(num_points*(.5 + size_no * .5))
        if(bot < 0) : 
            bot = 0
        if(top > 256) : 
            top = 256
        if(lft < 0) : 
            lft = 0
        if(rgt > 256) : 
            rgt = 256
        u[bot:top, lft:rgt] += 0.1
        v[bot:top, lft:rgt] += 0.5
        u = np.roll(u, (int((x_no - .5) * num_points), int((y_no - .5) * num_points)), axis=(0, 1))
        v = np.roll(v, (int((x_no - .5) * num_points), int((y_no - .5) * num_points)), axis=(0, 1))
        return u, v

    def three() :
        u = np.ones((num_points, num_points))
        v = np.zeros((num_points, num_points))
        return u, v

    def four() :
        u = np.ones((num_points, num_points))
        v = np.zeros((num_points, num_points))  
        x = np.linspace(-1, 1, num_points)
        y = np.linspace(-1, 1, num_points)
        X, Y = np.meshgrid(x, y)
        dR = np.exp(-36*np.power((np.power(X, shape_no*2) + np.power(Y, shape_no*2))/np.power(size_no, shape_no*2), sharpness))
        u += .1 * dR
        v += .5 * dR
        u = np.roll(u, (int((x_no - .5) * num_points), int((y_no - .5) * num_points)), axis=(0, 1))
        v = np.roll(v, (int((x_no - .5) * num_points), int((y_no - .5) * num_points)), axis=(0, 1))
        return u, v

    switcher = {
        1 : one(),
        2 : two(),
        3 : three(),
        4 : four()
    }

    return switcher.get(ic_no)

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