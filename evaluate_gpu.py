import scipy.io
import torch
import numpy as np
import os

#######################################################################
# Evaluate (Optimized for Recall@1 Only)
def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query).squeeze(1).cpu().numpy()

    # Get the top-1 prediction index (faster than full sorting)
    top_1_idx = np.argmax(score)  # Just get the index of the highest score

    # Check if the top-1 prediction is a correct match
    recall_1 = gl[top_1_idx] == ql

    return recall_1


######################################################################
# Load Data
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f']).cuda()
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
gallery_label = result['gallery_label'][0]

# Compute Recall@1
recall_1 = 0

for i in range(len(query_label)):
    recall_1 += evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)

# Convert to percentage
recall_1 = recall_1 / len(query_label) * 100

# Print result (same format as before)
print(f'Recall@1: {recall_1:.2f}')
