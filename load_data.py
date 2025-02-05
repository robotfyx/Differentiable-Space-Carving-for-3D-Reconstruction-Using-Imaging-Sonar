# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:29:55 2023
akutam_fyx
"""

import os
import pickle
import json
import math
from pyhocon import ConfigFactory
import numpy as np
from denoise.denoise import denoise

def load_data(conf, name):
    
    confname = 'confs/'+conf+'.conf'
    c = open(confname)
    conf_text = c.read()
    conf = ConfigFactory.parse_string(conf_text)
    x_max = conf.get_float('mesh.x_max')
    x_min = conf.get_float('mesh.x_min')
    y_max = conf.get_float('mesh.y_max')
    y_min = conf.get_float('mesh.y_min')
    z_max = conf.get_float('mesh.z_max')
    z_min = conf.get_float('mesh.z_min')
    object_bbox_min = conf.get_list('mesh.object_bbox_min')
    object_bbox_max = conf.get_list('mesh.object_bbox_max')
    
    cfg_pth = os.path.join('data/'+name, 'Config.json')
    pkl_pth = os.path.join('data/'+name, 'Data')
    
    with open(cfg_pth, 'r') as f:
        cfg = json.load(f)
    for agents in cfg["agents"][0]["sensors"]:
        if agents["sensor_type"] != "ImagingSonar": continue
        hfov = agents["configuration"]["Azimuth"]
        vfov = agents["configuration"]["Elevation"]
        min_range = agents["configuration"]["RangeMin"]
        max_range = agents["configuration"]["RangeMax"]
        hfov = math.radians(hfov)
        vfov = math.radians(vfov)
    
    images = []
    sensor_poses = []    
    
    for pkls in os.listdir(pkl_pth):
        filename = "{}/{}".format(pkl_pth, pkls)
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            image = state["ImagingSonar"]
            # s = image.shape
            # image[s[0]-180:,:] = 0
            
            pose = state["PoseSensor"]
            images.append(image)
            sensor_poses.append(pose)      
    
    images = np.array(images)
    print(f'{len(images)} images with noise')
    print('*'*20)
    images_no_noise = denoise(images)
    print(f'{len(images_no_noise)} images with no noise')
    data = {
        "images": images,
        "images_no_noise": images_no_noise, #[],#
        "sensor_poses": sensor_poses,
        "min_range": min_range,
        "max_range": max_range,
        "hfov": hfov,
        "vfov": vfov,
        "x_max": x_max,
        "x_min": x_min,
        "y_max": y_max,
        "y_min": y_min,
        "z_max": z_max,
        "z_min": z_min,
        "bbox_min": object_bbox_min,
        "bbox_max": object_bbox_max
    }
    
    i_train = np.arange(len(data["images"]))
    #np.random.shuffle(i_train)
    return data,i_train

if __name__ == '__main__':
    name = '14deg_planeFull'
    data,i_train = load_data(name)
    print(i_train[0:10])