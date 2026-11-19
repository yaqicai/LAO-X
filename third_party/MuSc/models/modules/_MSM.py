import torch
from tqdm import tqdm

"""
We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""

def resolve_topmin_interval(candidate_num, topmin_min=0, topmin_max=0.3):
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(candidate_num * k_max)
    if k_min < 1:
        k_min = int(candidate_num * k_min)
    k_max = max(1, min(int(k_max), candidate_num))
    k_min = max(0, min(int(k_min), candidate_num - 1))
    if k_max <= k_min:
        k_min = max(0, k_max - 1)
    return k_min, k_max

def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num-1, patch_num)
    patch2image = torch.min(patch2image, -1)[0]
    # interval average
    k_min, k_max = resolve_topmin_interval(patch2image.shape[1], topmin_min, topmin_max)
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_min, k_max = resolve_topmin_interval(patch2image.shape[1], topmin_min, topmin_max)
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def MSM(Z, device, topmin_min=0, topmin_max=0.3):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):
    # for i in range(Z.shape[0]):
        # anomaly_scores_i = compute_scores_fast(Z, i, device, topmin_min, topmin_max).unsqueeze(0)
        anomaly_scores_i = compute_scores_slow(Z, i, device, topmin_min, topmin_max).unsqueeze(0)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return anomaly_scores_matrix

if __name__ == "__main__":
    device = 'cuda:0'
    import time
    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time-s_time)*1000)
    
