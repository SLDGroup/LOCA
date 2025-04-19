import os
import torch
import matplotlib.pyplot as plt 
from utils.general_utils import AverageMeter
from os import path
from math import exp


def inverse_depth_norm(depth, maxDepth=10.0):
    valid_mask = depth != 0
    depth[valid_mask] = maxDepth / depth[valid_mask]
    zero_mask = depth == 0.0
    depth = torch.clamp(depth, maxDepth / 100, maxDepth)
    depth[zero_mask] = 0.0
    return depth

def compute_errors(gt, pred, nyuv2_test=False):
    """
    Compute error metrics between ground truth and predicted depth maps,
    excluding invalid pixels where ground truth depth is zero or negative.

    Args:
        gt (torch.Tensor): Ground truth depth map.
        pred (torch.Tensor): Predicted depth map.

    Returns:
        Tuple: Tuple containing various error metrics.
    """
    # Ensure tensors are float and on the same device
    device = gt.device
    gt = gt.float().to(device)
    if nyuv2_test:
        pred = inverse_depth_norm(depth=pred)
    else:
        pred = inverse_depth_norm(depth=pred)
        gt = inverse_depth_norm(depth=gt)
    pred = pred.float().to(device)

    # Create a mask of valid pixels (where gt > 0)
    valid_mask = (gt > 0) & (pred > 0)

    # Apply the valid mask to ground truth and predictions
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    # print("Pred",pred_valid.min(), pred_valid.max())
    # print("GT",gt_valid.min(), gt_valid.max(),"\n")

    # Compute threshold ratios for accuracy metrics
    thresh = torch.max(gt_valid / pred_valid, pred_valid / gt_valid)

    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    # Compute RMSE
    rmse = torch.sqrt(torch.mean((gt_valid - pred_valid) ** 2))

    # Compute RMSE of the logarithms
    rmse_log = torch.sqrt(torch.mean((torch.log(gt_valid) - torch.log(pred_valid)) ** 2))

    # Compute absolute relative difference
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)

    # Compute squared relative difference
    sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)

    return (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)