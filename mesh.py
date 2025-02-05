# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:15:10 2023
akutam_fyx
"""

import torch
from torch.nn import functional as F
import numpy as np
from render import net_output
import os
from skimage import measure
import trimesh
def plot_mesh(bbox_min, bbox_max, N, embed_fn, model, filename, i):
    with torch.no_grad():
        res_x = (bbox_max[0]-bbox_min[0])/(2*N-2)
        res_y = (bbox_max[1]-bbox_min[1])/(2*N-2)
        res_z = (bbox_max[2]-bbox_min[2])/(2*N-2)
        x = torch.linspace(bbox_min[0]+res_x, bbox_max[0]-res_x, N-1)
        y = torch.linspace(bbox_min[1]+res_y, bbox_max[1]-res_y, N-1)
        z = torch.linspace(bbox_min[2]+res_z, bbox_max[2]-res_z, N-1)
        X,Y,Z = torch.meshgrid((x,y,z))
        W = torch.stack((X,Y,Z), -1).reshape((N-1)**3,3)

        prob = net_output(W, embed_fn, model)    
        prob1 = prob.detach().cpu().numpy()
        np.savetxt(os.path.join(filename, f'prob_{i}.txt'), prob1)
        prob1 = prob1.reshape((N-1,N-1,N-1))
        bbox = np.array([bbox_min, bbox_max])
        voxel_size = list((bbox[1]-bbox[0]) / (np.array(prob1.shape)-1))
        # smoothed_data = gaussian_filter(prob1, sigma=1)
        verts, faces, normals, values = measure.marching_cubes(
            prob1, level=None, spacing=voxel_size
        )
        verts = verts+bbox_min
        mesh = trimesh.Trimesh(verts, faces, normals)
        name = os.path.join(filename, f'object_{i}.ply')
        mesh.export(name)
    
    return 0.5
