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
def load_network(name, opt):
    dirname = os.path.join('./model', name)
    last_model_name = get_model_list(dirname, 'net')  # Directly get the model file

    if not last_model_name:
        print("No model found!")
        return None, opt, None

    # Extract epoch number from filename
    match = re.search(r'net_(\d+|last)\.pth', last_model_name)
    epoch = int(match.group(1)) if match and match.group(1).isdigit() else match.group(1)

    # Load config directly without looping over keys
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        opt.__dict__.update(yaml.safe_load(stream))  # Faster way to update opt attributes

    # Initialize the model
    model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share)

    print(f'Loading model from {last_model_name}')
    model.load_state_dict(torch.load(last_model_name))

    return model, opt, epoch

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
