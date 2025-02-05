# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:12:25 2023
akutam_fyx
"""

import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

def get_arcs(H, W, phi_min, phi_max, r_min, r_max, c2w, n_selected_px, arc_n_samples, ray_n_samples, 
            hfov, px, r_increments, randomize_points, device, cube_center):
    
    i = px[:, 0]
    j = px[:, 1]

    # sample angle phi
    phi = torch.linspace(phi_min, phi_max, arc_n_samples).float().repeat(n_selected_px).reshape(n_selected_px, -1)

    dphi = (phi_max - phi_min) / arc_n_samples
    rnd = -dphi + torch.rand(n_selected_px, arc_n_samples)*2*dphi

    sonar_resolution = (r_max-r_min)/H
    if randomize_points:
        phi =  torch.clip(phi + rnd, min=phi_min, max=phi_max)

    # compute radius at each pixel
    r = i*sonar_resolution + r_min
    # compute bearing angle at each pixel
    theta = -hfov/2 + j*hfov/W
    # theta = np.pi/2-theta #训练实际数据变换了坐标轴，加上这句

    # Need to calculate coords to figure out the ray direction 
    # the following operations mimick the cartesian product between the two lists [r, theta] and phi
    # coords is of size: n_selected_px x n_arc_n_samples x 3
    coords = torch.stack((r.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1), 
                         theta.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1), 
                          phi), dim = -1)
    coords = coords.reshape(-1, 3)

    holder = torch.empty(n_selected_px, arc_n_samples*ray_n_samples, dtype=torch.long).to(device)
    bitmask = torch.zeros(ray_n_samples, dtype=torch.bool)
    bitmask[ray_n_samples - 1] = True
    bitmask = bitmask.repeat(arc_n_samples)


    for n_px in range(n_selected_px):
        holder[n_px, :] = torch.randint(0, i[n_px]-1, (arc_n_samples*ray_n_samples,))
        holder[n_px, bitmask] = i[n_px] 
    
    holder = holder.reshape(n_selected_px, arc_n_samples, ray_n_samples)
    
    holder, _ = torch.sort(holder, dim=-1)

    holder = holder.reshape(-1)
        

    r_samples = torch.index_select(r_increments, 0, holder).reshape(n_selected_px, 
                                                                    arc_n_samples, 
                                                                    ray_n_samples)
    
    rnd = torch.rand((n_selected_px, arc_n_samples, ray_n_samples))*sonar_resolution
    
    if randomize_points:
        r_samples = r_samples + rnd

    # rs = r_samples[:, :, -1]
    r_samples = r_samples.reshape(n_selected_px*arc_n_samples, ray_n_samples)

    theta_samples = coords[:, 1].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)
    phi_samples = coords[:, 2].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)

    # Note: r_samples is of size n_selected_px*arc_n_samples x ray_n_samples 
    # so each row of r_samples contain r values for points picked from the same ray (should have the same theta and phi values)
    # theta_samples is also of size  n_selected_px*arc_n_samples x ray_n_samples  
    # since all arc_n_samples x ray_n_samples  have the same value of theta, then the first n_selected_px rows have all the same value 
    # Finally phi_samples is  also of size  n_selected_px*arc_n_samples x ray_n_samples  
    # but not each ray has a different phi value
    
    # pts contain all points and is of size n_selected_px*arc_n_samples*ray_n_samples, 3 
    # the first ray_n_samples rows correspond to points along the same ray 
    # the first ray_n_samples*arc_n_samples row correspond to points along rays along the same arc 
    pts = torch.stack((r_samples, theta_samples, phi_samples), dim=-1).reshape(-1, 3)

# =============================================================================
#     dists = torch.diff(r_samples, dim=1)
#     dists = torch.cat([dists, torch.Tensor([sonar_resolution]).expand(dists[..., :1].shape).to(device)], -1)
# =============================================================================

    #r_samples_mid = r_samples + dists/2

    X_r_rand = pts[:,0]*torch.cos(pts[:,1])*torch.cos(pts[:,2])#中间一项，跑仿真数据x轴朝向物体，是cos；实际数据y轴朝向物体，是sin
    Y_r_rand = pts[:,0]*torch.sin(pts[:,1])*torch.cos(pts[:,2])
    Z_r_rand = pts[:,0]*torch.sin(pts[:,2])
    pts_r_rand = torch.stack((X_r_rand, Y_r_rand, Z_r_rand, torch.ones_like(X_r_rand)))


    pts_r_rand = torch.matmul(c2w, pts_r_rand)

    pts_r_rand = torch.stack((pts_r_rand[0,:], pts_r_rand[1,:], pts_r_rand[2,:]))

    # Centering step 
    pts_r_rand = pts_r_rand.T - cube_center

    # Transform to cartesian to apply pose transformation and get the direction
    # transformation as described in https://www.ri.cmu.edu/pub_files/2016/5/thuang_mastersthesis.pdf

    return dphi, r, pts_r_rand
    
def get_coords(target, ray_n_samples, percent_select_true, N_rand):
    h,w = target.shape
    # x,y = np.meshgrid(range(ray_n_samples,h), range(w))
    # coords_all = np.stack((x,y), axis=2).reshape(-1,2)#所有可采样像素点
    #按比例选择值较大的像素点
    true_index = np.where(target[ray_n_samples:h, :] >= 0.2)
    coords = np.stack(true_index, axis=1)
    
    
    #值较小的点选择0.5*N_rand个
    index = np.where((target[ray_n_samples:h, :] < 0.2) & (target[ray_n_samples:h, :] > 0))
    coords1 = np.stack(index, axis=1)
    
    
    #值为0的点选择0.5*N_rand个
    index1 = np.where(target[ray_n_samples:h, :] == 0.0)
    coords2 = np.stack(index1, axis=1)
    other_coords1 = coords2[np.random.choice(coords2.shape[0], int(N_rand/2))]
    
    #拼接起来
    if coords.shape[0] > 0 and coords1.shape[0] > 0:
        true_coords = coords[np.random.choice(coords.shape[0], int(percent_select_true*len(coords)))]
        other_coords = coords1[np.random.choice(coords1.shape[0], int(N_rand/2))]
        coords_all = np.concatenate((true_coords, other_coords, other_coords1))
    else:
        coords_all = other_coords1
    coords_all[:,0] += ray_n_samples
    coords_all = torch.from_numpy(coords_all).to(device)
    target = torch.Tensor(target).to(device)#将图像转为tensor移入GPU
    
    return coords_all,target