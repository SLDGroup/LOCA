import torch
import copy
import numpy as np


def aggregate_avg(local_weights, local_distr):
    """
    Receives a list of local model weights: local_weight
    And a list of local number of samples seen for the available clients: local_distr)
    """
    w_glob = None
    total_imgs = 0

    # N
    for nk in local_distr:
        total_imgs += nk

    for idx, client_weight in enumerate(local_weights):
        if idx == 0:
            w_glob = copy.deepcopy(client_weight)
            for k in w_glob.keys():
                w_glob[k] = torch.multiply(w_glob[k], local_distr[idx]/total_imgs)
        else:
            for k in w_glob.keys():
                w_glob[k] = torch.add(w_glob[k], torch.multiply(client_weight[k], local_distr[idx]/total_imgs))

    return w_glob

def compute_cos_coeffs(sigma, global_weights, local_weights):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    coeffs = {}
    for idx, w_local in enumerate(local_weights):
        if idx == 0:
            for k in global_weights.keys():
                if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    continue
                coeffs[k] = torch.exp(torch.mul(-sigma, cos(global_weights[k], w_local[k])))
        else:
            for k in global_weights.keys():
                if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    continue
                coeffs[k] = torch.add(coeffs[k], torch.exp(torch.mul(-sigma, cos(global_weights[k], w_local[k]))))
    return coeffs


def aggregate_cos(sigma, global_weights, local_weights, device):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    coeffs = compute_cos_coeffs(sigma=sigma, global_weights=global_weights, local_weights=local_weights)

    # Initialize w_glob as zeros, not using w_local[0]
    w_glob = {k: torch.zeros_like(global_weights[k]) for k in global_weights.keys()}

    # Loop over all local weights
    for idx, w_local in enumerate(local_weights):
        for k in global_weights.keys():
            if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                # For batch norm layers, average directly
                w_glob[k] = torch.add(w_glob[k], w_local[k])
            else:
                # Compute the weighting (beta) using the global weights and local weights
                beta = torch.div(torch.exp(torch.mul(-sigma, cos(global_weights[k], w_local[k]))), coeffs[k])
                w_glob[k] = torch.add(w_glob[k], torch.mul(w_local[k], beta.to(device)))

    # Normalize batch norm parameters
    for k in w_glob.keys():
        if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
            w_glob[k] = torch.div(w_glob[k], len(local_weights))

    return w_glob