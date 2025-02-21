import os
import torch
import yaml
from model import  three_view_net
import re 
import glob
from collections import Counter

def make_weights_for_balanced_classes(images, nclasses):
    count = Counter(item[1] for item in images)  # Faster counting
    N = float(sum(count.values()))
    
    weight_per_class = {cls: N / count[cls] for cls in count}  # Dictionary lookup (faster)
    
    weight = [weight_per_class[item[1]] for item in images]
    return weight




def get_model_list(dirname, key):
    if not os.path.exists(dirname):
        print(f'No directory found: {dirname}')
        return None
    
    model_files = glob.glob(os.path.join(dirname, f'*{key}*.pth'))
    
    return model_files[0] if model_files else None

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/'+dirname):
        os.mkdir('./model/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name, opt, device=None):
    
    model_path = "/home/gpu/Desktop/University1652-Baseline/model/three_view_long_share_d0.75_256_s1_google/net_119.pth"
    
    config_path = "/home/gpu/Desktop/University1652-Baseline/model/three_view_long_share_d0.75_256_s1_google/opts.yaml"
    with open(config_path, 'rb') as stream:
        opt.__dict__.update(yaml.safe_load(stream))

    # Initialize model
    model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share)

    print(f'Loading model from {model_path}...')

    # Set default device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model faster
    state_dict = torch.load(model_path, map_location='cpu')  # Load to CPU first
    model.load_state_dict(state_dict, strict=False)  # Disable strict mode if needed
    model = model.to(device)  # Move to GPU if available

    torch.cuda.empty_cache()  # Free unnecessary memory

    return model, opt
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def update_average(model_tgt, model_src, beta):
    for model in (model_src, model_tgt):
        toogle_grad(model, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_tgt.data.mul_(beta).add_(param_dict_src[p_name].data, alpha=(1. - beta))

    toogle_grad(model_src, True)
