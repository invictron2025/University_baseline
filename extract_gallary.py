# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
from torchvision import datasets,transforms
import os
import yaml
from utils import load_network
#####################################################################################################################################################################
parser = argparse.ArgumentParser(description="Extracting gallary features")
parser.add_argument('--name', default='/home/gpu/Desktop/University1652-Baseline/model/three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
opt = parser.parse_args()
#####################################################################################################################################################################
config_path = '/home/gpu/Desktop/University1652-Baseline/model/three_view_long_share_d0.75_256_s1_google/opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.CSafeLoader)
opt.fp16 = config['fp16'] 
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']
#####################################################################################################################################################################
# Function to get labels and paths from dataset
def get_id(img_path):
    labels = []
    paths = []
    for path, _ in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths
#####################################################################################################################################################################

# Load data
data_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dir = "/home/gpu/Desktop/Data/SpecificData/class_23"
dataset = datasets.ImageFolder(os.path.join(test_dir, "gallery_satellite"), data_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

#####################################################################################################################################################################

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

                outputs, _, _ = model(img_stack, None, None)
   
            # Sum flipped and original features in-place (reducing memory overhead)
            ff.copy_(outputs[:n] + outputs[n:])  

            # Normalize in-place for efficiency
            ff.div_(ff.norm(p=2, dim=1, keepdim=True) + 1e-12)  # Avoid NaNs

            # Append to feature list
            features.append(ff.cpu())  

    return torch.cat(features, dim=0)

#####################################################################################################################################################################

def save_gallery_features(model, dataloader, view_index, save_path):
    print("Extracting and saving gallery features...")

    # Extract features
    gallery_features = extract_feature(model, dataloader, view_index)

    # Get image labels and paths
    gallery_path = dataloader.dataset.imgs
    gallery_label, gallery_path = get_id(gallery_path)

    # Save to disk
    torch.save({
        'features': gallery_features,
        'labels': gallery_label,
        'paths': gallery_path
    }, save_path)

    print(f"Gallery features saved to {save_path}")

#####################################################################################################################################################################

# Load model
model, _ = load_network(opt.name, opt)
model.classifier.classifier = torch.nn.Sequential()
model = model.eval().cuda()

#####################################################################################################################################################################
if __name__ == "__main__":

    gallery_save_path = "gallery_features.pt"

    # Extract and save gallery features
    save_gallery_features(model, dataloader,1, gallery_save_path)
