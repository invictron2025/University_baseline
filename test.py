# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn

from torchvision import datasets,transforms
import time
import os
import scipy.io
import yaml
import math

from utils import load_network
from torch.amp import autocast



parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/gpu/Desktop/Data/AirStrip_Data',type=str, help='./test_data')
parser.add_argument('--name', default='/home/gpu/Desktop/University1652-Baseline/model/three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=3, type=int, help='views')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
opt.fp16 = config['fp16'] 
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

# gpu_ids = []
# for str_id in str_ids:
#     id = int(str_id)
#     if id >=0:
#         gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# # set gpu ids
# if len(gpu_ids)>0:
#     torch.cuda.set_device(gpu_ids[0])
#     cudnn.benchmark = True

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




data_dir = test_dir

num_workers = 16


image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite', 'query_drone']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                            shuffle=False, num_workers=num_workers,pin_memory=True) for x in ['gallery_satellite', 'query_drone']}
use_gpu = torch.cuda.is_available()


def which_view(name):
    view_mapping = {'satellite': 1, 'street': 2, 'drone': 3}
    for key, value in view_mapping.items():
        if key in name:
            return value
    print('unknown view')
    return -1




def extract_feature(model, dataloaders, view_index):
    features = []
    count = 0  # Track total processed images

    with torch.no_grad():  # Disable gradient calculation
        for img, label in dataloaders:
            img = img.cuda(non_blocking=True)  # Faster GPU transfer
            n = img.shape[0]
            count += n
            print(f"Processed: {count}")

            # Pre-allocate feature tensor with the correct size
            ff = torch.empty(n, 512, device="cuda")

            # Create flipped images and stack
            img_flipped = torch.flip(img, dims=[3])  # Flip horizontally
            img_stack = torch.cat((img, img_flipped), dim=0)  # Avoid list-based cat

            with torch.autocast("cuda"):  # Just use this, no arguments needed

                if view_index == 1:
                    outputs, _, _ = model(img_stack, None, None)
                elif view_index == 3:
                    _, _, outputs = model(None, None, img_stack)
                else:
                    raise ValueError(f"Invalid view_index: {view_index}")  # Avoid silent failures

            # Sum flipped and original features in-place (reducing memory overhead)
            ff.copy_(outputs[:n] + outputs[n:])  

            # Normalize in-place for efficiency
            ff.div_(ff.norm(p=2, dim=1, keepdim=True) + 1e-12)  # Avoid NaNs

            # Append to feature list
            features.append(ff.cpu())  

    return torch.cat(features, dim=0)


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, = load_network(opt.name, opt)
model.classifier.classifier = nn.Sequential()
model = model.eval().cuda()

# Extract feature
since = time.time()

gallery_name = 'gallery_satellite'
query_name = 'query_drone'


which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))

gallery_path = image_datasets[gallery_name].imgs

query_path = image_datasets[query_name].imgs


gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    with torch.no_grad():
        query_feature = extract_feature(model,dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model,dataloaders[gallery_name], which_gallery)

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature.numpy(),'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat('pytorch_result.mat',result)

    # print(opt.name)
    result = './model/%s/result.txt'%opt.name
    os.system('python demo.py | tee -a %s'%result)
    os.system('python evaluate_gpu.py | tee -a %s'%result)