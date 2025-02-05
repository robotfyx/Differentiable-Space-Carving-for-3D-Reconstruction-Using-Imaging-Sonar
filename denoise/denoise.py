# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:21:04 2023

@author: Administrator
"""

import cv2
import torch
import numpy as np
import pickle
from tqdm import trange

def uint2single(img):

    return np.float32(img/255.)
def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    return img#np.uint8((img*255.0).round())#
# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

n_channels = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './denoise/scunet_gray_25.pth'
from .network_scunet import SCUNet as net
# model_path = '/home/kemove/DSC_Code/denoise/scunet_gray_25.pth'
# from network_scunet import SCUNet as net
model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

def denoise(images):
    
    images_no_noise = list()
        
    for i in trange(len(images)):
        images[i][images[i]>1] = 1
        img = images[i]
        
        img_L = np.expand_dims(img, axis=2)#
        img_L = single2tensor4(img_L)
        img_L = img_L.to(device)
        img_E = model(img_L)#N 1 W H
        img_E = tensor2uint(img_E)#N W H
        
        if i >= 1:
            img_E = img_E-images_no_noise[0]#减去背景图
            img_E[img_E<0] = 0
            img_E[img_E<1e-2] = 0
            
            # new = np.zeros_like(img)
            # new[img_E>0] = img[img_E>0]
            # new[new<0.2] = 0
            # images_no_noise.append(new)
            images_no_noise.append(img_E)
        else:
            images_no_noise.append(img_E)
        
        del(img_L)
    images_no_noise[0][:,:] = 0
    return images_no_noise
    
if __name__ =='__main__':
    with open('/home/kemove/DSC_Code/data/ship0/Data/111.pkl', 'rb') as f:
        data1 = pickle.load(f)
    img1 = data1["ImagingSonar"]
    with open('/home/kemove/DSC_Code/data/ship0/Data/1.pkl', 'rb') as f:
        data = pickle.load(f)
    img = data["ImagingSonar"]#[:707,:]
    # cv2.imwrite('noised images.png', np.uint8((img1*255.0).round()))
    images = np.array([img, img1])#N H W
    out = denoise(images)
    
    # out_img = out[1]
    # out_img[out_img>0.2] = 0.85
    # print(len(np.where(out[1]>0.1)[0]))
    # cv2.imwrite('E:/pictures for DSC/pipeline/probability_2.png', np.uint8((out_img*255.0).round()))
    cv2.imshow('original', img1)
    cv2.imshow('output', out[1])

    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    