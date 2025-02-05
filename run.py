# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:45:31 2023
akutam_fyx
"""

import configargparse
import numpy as np
import torch
from embed import get_embedder
from network import network
from load_data import load_data
from radam import RAdam
from get_random_coords import get_arcs,get_coords
from render import render_sonar
from loss import total_variation_loss
from tqdm import trange
from mesh import plot_mesh
import os
import time
import torch
from matplotlib import pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(30)

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='14deg_planeFull', help='the config file')
    parser.add_argument('--name', type=str, default='14deg_planeFull')
    
    parser.add_argument("--finest_res",   type=int, default=128,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=14,
                        help='log2 of hashmap size')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--tv_loss_weight", type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument("--hash_n_level", type=int, default=8,
                        help='number of resolutions')
    parser.add_argument("--test_res", type=int, default=128,
                        help='the resolution used when reconstructing the object')
    parser.add_argument("--expname", type=str, default='test')
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--decay_step", type=int, default=1000)
    parser.add_argument("--percent", type=float, default=0.25)
    parser.add_argument("--ray_n_samples", type=int, default=64)
    parser.add_argument("--arc_n_samples", type=int, default=10)
    parser.add_argument("--sparsity_loss_weight", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--N_rand", type=int, default=100)
    
    return parser


def create(args):
    embed_fn, input_ch = get_embedder(args)#哈希编码函数，密度网络输入维度
    embedding_params = list(embed_fn.parameters())#可训练参数  
    model = network(input_ch=input_ch).to(device)
    grad_vars = list(model.parameters())#可训练参数
    optimizer = RAdam([
                        {'params': grad_vars, 'weight_decay': 1e-6},
                        {'params': embedding_params, 'eps': 1e-15}
                    ], lr=args.lrate, betas=(0.9, 0.99))
    
    criterion = torch.nn.L1Loss(reduction='mean')#L1Loss
    
    return embed_fn,model,optimizer,criterion#,bd

def save_checkpoints(embed_fn, model, optimizer, path):
    checkpoints = {
        'embed_fn':embed_fn.state_dict(),
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict()
        }
    torch.save(checkpoints, path)
    print('Successfully saved checkpoints!')
    
def process_target(target):
    target[target>0.2] = torch.sigmoid(7*target[target>0.2])
    return target

def train(args):
    #加载数据
    data,i_train = load_data(args.conf, args.name)
    
    #获取一些数据
    H,W  = data['images'][0].shape
    phi_min = -data["vfov"]/2
    phi_max = data["vfov"]/2
    r_min = data["min_range"]
    r_max = data["max_range"]
    x_max = data['x_max']
    x_min = data['x_min']
    y_max = data['y_max']
    y_min = data['y_min']
    z_max = data['z_max']
    z_min = data['z_min']
    args.bounding_box = [torch.tensor([x_min-(x_max + x_min)/2,y_min-(y_max + y_min)/2,z_min-(z_max + z_min)/2]), \
                         torch.tensor([x_max-(x_max + x_min)/2,y_max-(y_max + y_min)/2,z_max-(z_max + z_min)/2])]
    cube_center = torch.Tensor([(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2])
    
    bbox_min = data['bbox_min']
    bbox_max = data['bbox_max']
    args.bbox = [bbox_min, bbox_max]
    hfov = data['hfov']
    
    r_increments = []
    sonar_resolution = (r_max-r_min)/H
    for i in range(H):
        r_increments.append(i*sonar_resolution + r_min)

    r_increments = torch.FloatTensor(r_increments).to(device)

    
    #建立模型
    embed_fn,model,optimizer,criterion = create(args)
    
    global_step = 0
    
    os.makedirs(os.path.join('results', args.expname), exist_ok=True)
    f = os.path.join(os.path.join('results', args.expname), 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
        file.close()
    
    Niters = len(i_train)
    with open(f, 'a') as file:
        file.write('\n')
        file.write('Niters={}'.format(Niters))
        file.close()
    times = 0
    
    
    l_total = list()
    l_tv = list()
    l_weight = list()
    l_sparsity = list()
    l_norm = list()
    
    os.makedirs(os.path.join('results', args.expname, 'loss'), exist_ok=True)
    os.makedirs(os.path.join('results', args.expname, 'checkpoints'), exist_ok=True)        
    
    print('Start training:')
    for i in range(args.iters):
        print('step:',i)
        np.random.shuffle(i_train)
        loss_total = 0.0
        loss_tv = 0.0
        loss_weight = 0.0
        loss_sparsity = 0.0
        loss_norm = 0.0
        
        # tv_loss_list = list()
        # weight_loss_list = list()
        
        start_time = time.time()
        for j in trange(0, Niters):
            target = data['images_no_noise'][i_train[j]]#目标图像
            pose = data['sensor_poses'][i_train[j]]#该张图位姿
            
            coords,target = get_coords(target, ray_n_samples=args.ray_n_samples, percent_select_true=args.percent, N_rand=args.N_rand)#获取采样像素点和图像张量    
            n_pixels = len(coords)
            
            _, _, pts = get_arcs(H, W, phi_min, phi_max, r_min, r_max, 
                                                       torch.Tensor(pose).to(device), n_pixels, args.arc_n_samples, args.ray_n_samples, 
                                                       hfov, coords, r_increments, True, device, cube_center.to(device))
            
            target_s = target[coords[:, 0], coords[:, 1]]
            pts.requires_grad_(True)
            add_norm_loss = True if i >= 1 else False#从第二轮训练开始加进norm loss
            # target_01 = target_s>0#.float()
            w,sparsity,norm_loss = render_sonar(pts, embed_fn, model, n_pixels, \
                               arc_n_samples=args.arc_n_samples, ray_n_samples=args.ray_n_samples, \
                               add_norm_loss=add_norm_loss)
            
            #weight loss
            
            weight_loss = criterion(process_target(target_s), w)#+criterion(process_target(target_s), sw)#criterion(target_s, w)#

            sparsity_loss_weight = 1e-6# if i ==0 else 1e-8
            loss = weight_loss+sparsity_loss_weight*sparsity+1e-2*norm_loss
             
            #TV loss
            TV_loss = sum(total_variation_loss(embed_fn.embeddings[k], 16, args.finest_res, k, args.log2_hashmap_size, args.hash_n_level) for k in range(args.hash_n_level))
            loss = loss + args.tv_loss_weight * TV_loss
            # if global_step>1000:
            #     args.tv_loss_weight = 0.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                loss_tv += TV_loss
                loss_total += loss
                loss_weight += weight_loss
                loss_sparsity += sparsity
                loss_norm += norm_loss
                
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * args.decay_step
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
                       
            global_step += 1
            
            #清除变量，释放显存
            del(target)
            del(target_s)
            del(pts)
            del(w)
            del(coords)
           
        end_time = time.time()
        spend_time = end_time-start_time
        times += spend_time
        
        l_total.append(loss_total.detach().cpu()/len(i_train))
        l_tv.append(loss_tv.detach().cpu()/len(i_train))
        l_weight.append(loss_weight.detach().cpu()/len(i_train))
        l_sparsity.append(loss_sparsity.detach().cpu()/len(i_train))
        l_norm.append(loss_norm.detach().cpu()/len(i_train))
        
        with torch.no_grad():
            l = loss_total/len(i_train)
            lnorm = loss_norm/len(i_train)
            lsparsity = loss_sparsity/len(i_train)
                      
        level = plot_mesh(bbox_min, bbox_max, args.test_res, embed_fn, model, \
                  os.path.join('results', args.expname), i)
        
        with open(f, 'a') as file:
            file.write('\n')
            file.write('level_{}={}'.format(i, level))
            file.close()
        print(f'total loss:{l},sparsity:{lsparsity},norm error:{lnorm}')


        if l < 0.5:
            save_checkpoints(embed_fn, model, optimizer, \
                             os.path.join('results', args.expname, 'checkpoints', 'checkpoints_{}.pth'.format(i)))
        
        torch.cuda.empty_cache()
        
    times1 = times/args.iters
    with open(f, 'a') as file:
        file.write('\n')
        file.write('spend time={} min {} s per training round\n'.format(int(times1/60), times1%60))
        file.write('total time={} min {} s'.format(int(times/60), times%60))
        file.close()
    
    np.savetxt(os.path.join('results', args.expname, 'loss', 'l_total.txt'), l_total)
    np.savetxt(os.path.join('results', args.expname, 'loss', 'l_tv.txt'), l_tv)
    np.savetxt(os.path.join('results', args.expname, 'loss', 'l_weight.txt'), l_weight)
    np.savetxt(os.path.join('results', args.expname, 'loss', 'l_sparsity.txt'), l_sparsity)
    np.savetxt(os.path.join('results', args.expname, 'loss', 'l_norm.txt'), l_norm)
    
if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    
    parser = config_parser()
    args = parser.parse_args()
    
    train(args)
    
    