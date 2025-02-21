# match_query.py
import torch
import argparse
import os
from torchvision import datasets, transforms
from utils import load_network  
import yaml
import pickle
#####################################################################################################################################################################
parser = argparse.ArgumentParser(description="Query Image Matching")
parser.add_argument("--query_index", default=0, type=int, help="Index of the query image")
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
# Load data
data_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dir = "/home/gpu/Desktop/Data/SpecificData/class_23"
dataset = datasets.ImageFolder(os.path.join(test_dir, "query_drone"), data_transforms)
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

                _, _, outputs = model(None, None, img_stack)
         
            # Sum flipped and original features in-place (reducing memory overhead)
            ff.copy_(outputs[:n] + outputs[n:])  

            # Normalize in-place for efficiency
            ff.div_(ff.norm(p=2, dim=1, keepdim=True) + 1e-12)  # Avoid NaNs

            # Append to feature list
            features.append(ff.cpu())  

    return torch.cat(features, dim=0)
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
def match_query(query_index, model, dataloader, fp16):
    print(f"Matching query index {query_index}...")

    # Load precomputed gallery features
    with open("gallery_features.pt", "rb") as f:
        gallery_data = pickle.load(f, mmap_mode="r")
    gallery_features = gallery_data['features']
    gallery_labels = gallery_data['labels']
    gallery_paths = gallery_data['paths']

    # Extract query features
    query_feature = extract_feature(model, dataloader, 3)

    # Get query labels
    query_path = dataloader.dataset.imgs
    query_labels, _ = get_id(query_path)  

    # Matching process
    query_vector = query_feature[query_index].view(-1, 1)  

    # Ensure dtype consistency
   
    gallery_features = gallery_features.half()  # Convert to float16
    query_vector = query_vector.half()  # Convert to float16


    scores = torch.mm(gallery_features, query_vector).squeeze(1)  # Cosine similarity
    index = torch.topk(scores, k=10, largest=True).indices.cpu().numpy()  # Get top matches

    # Check if the top match is correct
    if gallery_labels[index[0]] == query_labels[query_index]:
        print(f"Correct Match: {gallery_labels[index[0]]}")
    else:
        correct_match = None
        wrong_match = index[0]

        for idx in index[1:]:  
            if gallery_labels[idx] == query_labels[query_index]:
                correct_match = idx
                print(f"Correct Match: {gallery_labels[correct_match]}")
                break

        if correct_match is None:
            print(f"Wrong Match: {gallery_labels[wrong_match]}")
#####################################################################################################################################################################
# Load model
model, _ = load_network(opt.name, opt)
model.classifier.classifier = torch.nn.Sequential()
model = model.eval().cuda()

#####################################################################################################################################################################

if __name__ == "__main__":
    # matching the query features
    match_queryy=match_query(opt.query_index, model, dataloader, opt.fp16)
