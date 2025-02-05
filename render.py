# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:14:28 2023
akutam_fyx
"""

import torch

from torch.nn import functional as F

def net_output(pts, embed_fn, model):
    
    #通过编码获得输入
    prod_input,keep_mask = embed_fn(pts)
    
    out = model(prod_input)
    out[~keep_mask] = 0
    
    return out

def random_point_on_plane(point, normal, max_distance=0.2):
    # 生成随机向量
    random_vector = torch.randn_like(normal)
    # 归一化为单位向量
    random_vector = random_vector / torch.norm(random_vector)
    # 叉乘得到与法向量垂直的向量
    perpendicular_vector = torch.cross(random_vector, normal)
    # 计算新点的坐标
    new_point = point + perpendicular_vector * max_distance
    return new_point

def render_sonar(
        pts,
        embed_fn,
        model,
        n_pixels,
        arc_n_samples,
        ray_n_samples,
        add_norm_loss
        ):   
    prob = net_output(pts, embed_fn, model)#得到每个采样点预测的prod值    
    prob1 = prob.reshape(n_pixels, arc_n_samples, ray_n_samples)
    rayprob = prob1[:,:,:ray_n_samples-2]#每条射线上前ray_n_samples-1个点的prob，用于计算T
    Transmit = torch.prod(1-rayprob, dim=-1)#T
    probOnArc = prob1[:,:,-1]#弧上的点的prob
    pixprob = Transmit*probOnArc#弧上点被看到的概率
    w = torch.prod(1-pixprob, dim=1)#像素不存在概率
    
    error = torch.sum(prob)#+torch.sum(sprob)
    
    if add_norm_loss:
        surf_mask = (prob > 0.1).squeeze(1)#存在概率大于0.1的点取为表面点
        surf_points = pts[surf_mask]#用mask将表面点取出
        surfN = surf_points.shape[0]
        surf_points_neig = surf_points + (torch.rand_like(surf_points) - 0.5) * 0.1#每个表面点取一个邻域内的点
        #求梯度
        pp = torch.cat([surf_points, surf_points_neig], dim=0)#将表面点与邻域点连接起来
        y = net_output(pp, embed_fn, model)#传进网络
        with torch.enable_grad():
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            g = torch.autograd.grad(
                outputs = y,
                inputs = pp,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, 
                allow_unused=True
                )[0]
        normals = g/(g.norm(2, dim=1, keepdim=True)+1e-10)
        # assert not torch.isnan(normals).any(), "normals contains nan values!"
        diff_norm = torch.norm(normals[:surfN]-normals[surfN:],dim=-1)
        # assert not torch.isnan(diff_norm).any(), "diff_norm contains nan values!"
        # assert diff_norm.shape[0] > 0, "diff_norm is empty!"
        if diff_norm.shape[0] == 0:
            norm_error = torch.tensor(0.0).cuda().float()
        else:
            norm_error = torch.mean(diff_norm)
        return 1-w,error,norm_error    
    # elif add_norm_prob:
    #     surf_mask = prob > 0.1#存在概率大于0.1的点取为表面点       
    #     surf_prob = prob[surf_mask]#将表面点预测的概率取出
    #     #求梯度
    #     with torch.enable_grad():
    #         d_output = torch.ones_like(surf_prob, requires_grad=False, device=surf_prob.device)
    #         g = torch.autograd.grad(
    #             outputs = surf_prob,
    #             inputs = pts,
    #             grad_outputs=d_output,
    #             create_graph=True,
    #             retain_graph=True,
    #             only_inputs=True, 
    #             allow_unused=True
    #             )[0]
    #     ssurf_mask = torch.norm(g, dim=-1) > 0.1#将梯度模长较大的取出，这些是真正的边界点
    #     surf_g = g[ssurf_mask]
    #     surf_points = pts[ssurf_mask]#用mask将表面点取出
    #     normals = surf_g/(surf_g.norm(2, dim=1, keepdim=True)+1e-10)#归一化
    #     new_pts = surf_points+0.1*normals#沿梯度方向距离0.1取一个点
    #     new_probs = net_output(new_pts, embed_fn, model)
    #     if new_probs.shape[0] == 0:
    #         norm_prob_error = torch.tensor(0)
    #     else:
    #         norm_prob_error = torch.mean(torch.abs(prob[ssurf_mask]-new_probs))
    #     return 1-w,error,norm_prob_error
    
    else:
        return 1-w,error,torch.tensor(0)

    
    