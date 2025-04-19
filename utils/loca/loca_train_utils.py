import torch
import copy
from utils.dataset_utils import load_data
from utils.general_utils import AverageMeter
from utils.skipping_utils import SSIM
from utils.fl_utils import aggregate_cos
from utils.loss_utils import Depth_Loss
import random


def compute_dynamic_thresholds(current_frame, l="l1", base_factor=0.01):
    """
    Compute dynamic L1 and L2 norm thresholds based on the mean and standard deviation
    of pixel intensities in the current frame.

    Args:
        current_frame (torch.Tensor): The current frame tensor of shape (C, H, W) with pixel values in [0, 1].
        base_factor (float): A scaling factor for adjusting the sensitivity of the threshold.

    Returns:
        l1_threshold (float): Computed L1 norm threshold.
        l2_threshold (float): Computed L2 norm threshold.
    """
    # Compute mean and standard deviation of the pixel values in the current frame
    if l == "l1":
        mean_pixel_intensity = torch.mean(current_frame).item()
        threshold = base_factor * mean_pixel_intensity * current_frame.numel()  # Scale by the number of elements (H x W x C)
    elif l == "l2":
        std_pixel_intensity = torch.std(current_frame).item()
        threshold = base_factor * std_pixel_intensity * torch.sqrt(torch.tensor(current_frame.numel())).item()  # Scale by sqrt(numel)
    return threshold


# Function to perform center or random cropping
def crop_image(image, crop_size, top=None, left=None):
    """Crop the image based on the specified size and position."""
    _, _, h, w = image.shape
    crop_h, crop_w = crop_size

    # If top and left are not provided, calculate center crop as default
    if top is None or left is None:
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

    # Return the cropped image
    return image[:, :, top:top + crop_h, left:left + crop_w]


def train_loca(model, model_path, cuda_name, optimizer, k1, kt, dev_idx, dataset_name, seed, batch_size,
               train_type=None, lr=None, num_rooms="3", data_iid=False):
    device = torch.device(cuda_name)

    train_loaders, test_loaders = load_data(test_global=False, dev_idx=dev_idx, dataset_name=dataset_name, seed=seed,
                                            batch_size=batch_size, train_type=train_type, train_split=0.8,
                                            num_rooms=num_rooms, data_iid=data_iid)

    model.train()
    depth_loss = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)
    train_loss = AverageMeter()

    ssim_eval = SSIM(data_range=1.0, size_average=True)
    skip_thresh = 0.4
    crop_size = (200,200)
    random_crop = True

    global_weights = torch.load(model_path, map_location=device)

    for k1_idx in range(k1):
        deltas = [AverageMeter() for _ in range(3)]
        for loader_idx in range(len(train_loaders)):
            loader = train_loaders[loader_idx]
            model.train()
            for kt_idx in range(kt):
                for batch_idx, batch in enumerate(loader):
                    _, _, img, depth = batch
                    if img.size(0) == 1:
                        continue
                    img = img.to(device, non_blocking=True)
                    depth = depth.to(device, non_blocking=True)

                    # Batch skipping comparing the first images
                    if batch_idx > 0:
                        # Values for SSIM eval over entire batch
                        current_frame = batch[0][0].unsqueeze(0).to(device, non_blocking=True)
                        prev_frame = prev_saved_frame.unsqueeze(0).to(device, non_blocking=True)

                        if random_crop:
                            # Generate a new random crop position for this batch
                            _, _, h, w = current_frame.shape
                            top = random.randint(0, h - crop_size[0])
                            left = random.randint(0, w - crop_size[1])
                            prev_crop_position = (top, left)  # Store the crop position
                        else:
                            _, _, h, w = current_frame.shape
                            # Use the same position as the center crop
                            top, left = (h - crop_size[0]) // 2, (w - crop_size[1]) // 2
                            prev_crop_position = (top, left)  # Store the crop position

                            # Apply the same cropping to both current and previous frames
                        current_frame_cropped = crop_image(current_frame, crop_size, top=prev_crop_position[0],
                                                           left=prev_crop_position[1])
                        prev_frame_cropped = crop_image(prev_frame, crop_size, top=prev_crop_position[0],
                                                        left=prev_crop_position[1])

                        # Compute the similarity metric
                        score = ssim_eval(current_frame_cropped, prev_frame_cropped)

                        # Skip the batch if the norm difference or SSIM score is above the skip threshold
                        if score >= skip_thresh:
                            # print("batch skipped")
                            continue

                    optimizer.zero_grad()
                    pred = model(img)
                    loss = depth_loss(pred, depth)

                    train_loss.update(loss.item(), img.size(0))
                    loss.backward()
                    optimizer.step()
                    prev_saved_frame = batch[0][0].to(device, non_blocking=True)

            w_seq = model.state_dict()
            if loader_idx == 0:
                local_weights = []
            local_weights.append(copy.deepcopy(w_seq))

            w_avg = aggregate_cos(sigma=0.1, global_weights=global_weights, local_weights=local_weights,
                                  device=device)
            model.load_state_dict(w_avg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    torch.save(model.state_dict(), model_path)

    # print(f"Dev{dev_idx}: train loss={train_loss.avg:.6f}")
    return train_loss.avg, deltas



