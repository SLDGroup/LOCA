import torch
import torch.nn.functional as F
from math import exp
import numpy as np


class Depth_Loss():
    def __init__(self, alpha, beta, gamma, maxDepth=10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.maxDepth = maxDepth

        self.L1_Loss = torch.nn.L1Loss()


    def __call__(self, output, depth):
        valid_mask = (depth > 0.0).float()  # Shape: [batch_size, 1, height, width]

        # Apply the valid mask
        output_masked = output * valid_mask
        depth_masked = depth * valid_mask

        # Compute L1 Loss
        if valid_mask.sum() > 0:
            l_depth = self.L1_Loss(output_masked, depth_masked)
            # Compute SSIM Loss
            l_ssim = torch.clamp((1 - self.ssim(output_masked, depth_masked, valid_mask, self.maxDepth)) * 0.5, 0, 1)
            # Compute Gradient Loss
            l_grad = self.gradient_loss(output_masked, depth_masked, valid_mask)
        else:
            l_depth = torch.tensor(0.0, device=output.device)
            l_ssim = torch.tensor(0.0, device=output.device)
            l_grad = torch.tensor(0.0, device=output.device)

        # Combine the losses
        loss = self.alpha * l_depth + self.beta * l_ssim + self.gamma * l_grad

        return loss

    def ssim(self, img1, img2, mask, val_range, window_size=11, window=None, size_average=True, full=False):
        L = val_range

        # Apply the mask
        img1 = img1 * mask
        img2 = img2 * mask

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            padd = window_size // 2

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret

    def gradient_loss(self, gen_frames, gt_frames, mask, alpha=1):
        # Apply the mask to the input frames
        gen_frames_masked = gen_frames * mask
        gt_frames_masked = gt_frames * mask

        # Compute gradients
        gen_dx, gen_dy = self.gradient(gen_frames_masked)
        gt_dx, gt_dy = self.gradient(gt_frames_masked)

        # Compute gradient differences
        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # Apply the mask to gradient differences (mask dimensions match grad_diff_x and grad_diff_y)
        grad_mask = mask  # No slicing needed
        grad_diff_x = grad_diff_x * grad_mask
        grad_diff_y = grad_diff_y * grad_mask

        # Compute combined gradient loss
        grad_comb = grad_diff_x ** alpha + grad_diff_y ** alpha

        # Avoid division by zero
        valid_pixels = grad_mask.sum()
        if valid_pixels > 0:
            loss = grad_comb.sum() / valid_pixels
        else:
            loss = torch.tensor(0.0, device=gen_frames.device)

        return loss

    def gradient(self, x):
        """
        idea from tf.image.image_gradients(image)
        https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        """

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top

        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()