import argparse
import scipy.io
import torch
import numpy as np
import os

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=4, type=int, help='test_image_index')
parser.add_argument('--test_dir', default='/home/gpu/Desktop/Data/Test_Single', type=str, help='./test_data')
opts = parser.parse_args()

######################################################################
# Load Features and Labels
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.tensor(result['query_f'], dtype=torch.float32, device="cuda")
query_label = result['query_label'][0]
gallery_feature = torch.tensor(result['gallery_f'], dtype=torch.float32, device="cuda")
gallery_label = result['gallery_label'][0]

#######################################################################
# Compute similarity & sort
i = opts.query_index
query = query_feature[i].view(-1, 1)  # Reshape for matrix multiplication
scores = torch.mm(gallery_feature, query).squeeze(1)  # Compute cosine similarity
index = torch.argsort(scores, descending=True).cpu().numpy()  # Sort in descending order

#######################################################################
# Print Result
top_match_label = gallery_label[index[0]]
if top_match_label == query_label[i]:
    print(f"Correct Match: Label {top_match_label}")
else:
    print(f"Wrong Match: Label {top_match_label}")
