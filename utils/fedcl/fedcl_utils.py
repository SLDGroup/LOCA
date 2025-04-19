import torch

from utils.dataset_utils import load_data, get_transforms
from utils.general_utils import AverageMeter
from math import exp
from copy import deepcopy
from utils.loss_utils import Depth_Loss
from utils.testing_utils import compute_errors
import copy
import os
from models.get_model import get_model


def train_fedcl(model, dev_idx, model_path, cuda_name, optimizer, local_epochs, N_fcl, coe, dataset_name, seed,
                kt, comm_round=-1, batch_size=20, num_rooms="3", data_iid=False):
    device = torch.device(cuda_name)

    train_loaders, test_loaders = load_data(test_global=False, dev_idx=dev_idx, dataset_name=dataset_name, seed=seed,
                                            batch_size=batch_size, train_type="fedcl", train_split=0.8,
                                            num_rooms=num_rooms, data_iid=data_iid)

    depth_loss = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)

    # task_metrics = {'fedcl': list()}
    # num_tasks = len(data_loaders)

    if (comm_round+1) % N_fcl == 0:
        global_weights = torch.load(os.path.join(os.path.dirname(model_path), f"global_weights.pth"), map_location=device)
        for key, tensor in global_weights.items():
            # Check if the tensor is of a floating-point type
            if torch.is_floating_point(tensor):
                global_weights[key] = tensor.to(device).detach().requires_grad_(True)
            else:
                # Non-floating-point tensors cannot have requires_grad=True, just move to device
                global_weights[key] = tensor.to(device)
        w_d = torch.load(os.path.join(os.path.dirname(model_path), f"comm_w_d.pth"), map_location=device)

    train_loss = AverageMeter()
    deltas = [AverageMeter() for _ in range(3)]
    model.train()
    for epoch_idx in range(local_epochs):
        for loader_idx in range(len(train_loaders)):
            loader = train_loaders[loader_idx]
            for kt_idx in range(kt):
                for batch_idx, batch in enumerate(loader):
                    _, _, image, depth = batch
                    if image.size(0) == 1:
                        continue
                    image = image.to(device, non_blocking=True)
                    depth = depth.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    pred = model(image)

                    loss = depth_loss(pred, depth)

                    # print(comm_round, N_fcl, (comm_round + 1) % N_fcl == 0)
                    if (comm_round+1) % N_fcl == 0:
                        cons_loss = 0
                        for name, param in model.named_parameters():
                            cons_loss += w_d[name].to(device) * torch.sum((global_weights[name] - param) ** 2)
                        cons_loss *= coe
                        loss = loss + cons_loss

                    train_loss.update(loss.item(), image.size(0))
                    loss.backward()
                    optimizer.step()
            # print(f"Device {dev_idx}, loss: {train_loss.avg}")


    torch.save(model.state_dict(), model_path)
    return train_loss.avg, deltas


def ewc(model, cuda_name, dataset_name, seed, train_type, epochs=1, learning_rate=1e-4, batch_size=32, num_rooms="3", data_iid=False):
    device = torch.device(cuda_name)
    model = model.to(device)
    proxy_loader = load_data(train_type=train_type, seed=seed, dataset_name=dataset_name, test_global=False,
                             batch_size=batch_size, num_rooms=num_rooms, data_iid=data_iid)

    criterion = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # tmp_weights = copy.deepcopy(model.state_dict())
    # for k in tmp_weights.keys():
    #     tmp_weights[k] = torch.zeros_like(tmp_weights[k])
    tmp_weights = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

    model.eval()
    num_examples = 0
    for _ in range(epochs):
        for idx, batch in enumerate(proxy_loader):
            # if idx >= 5:
            #     continue
            images, depth = batch[2], batch[3]
            images, depth = images.to(device), depth.to(device)
            num_examples += images.size(0)

            # compute output
            pred = model(images)
            loss = criterion(pred, depth)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            for k, p in model.named_parameters():
                tmp_weights[k].add_(p.grad.detach() ** 2)

        for k, v in tmp_weights.items():
            tmp_weights[k] = torch.sum(v).div(num_examples)

    return tmp_weights