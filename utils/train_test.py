import time
import torch
from models.get_model import get_model
from utils.dataset_utils import load_data
from utils.general_utils import AverageMeter
from utils.fedcl.fedcl_utils import train_fedcl
from utils.testing_utils import compute_errors
from utils.fedknow.fedknow_utils import train_fedknow
from utils.loca.loca_train_utils import train_loca
from utils.codeps.codeps_train_utils import train_codeps
from utils.loss_utils import Depth_Loss


def global_test(model, dataset_name, seed, device=None, cuda_name=None, num_rooms="3", nyuv2_test=False):
    """
    General test function used by the cloud for evaluation

    model (torch.nn.Module): PyTorch model with loaded weights to evaluate on
    cuda_name (str): CUDA device selection
    dataset_name (str): Only NYUv2
    data_iid (bool): Desired IID/NIID split
    seed (int): Random seed
    verbose (bool): Enable/disable debugging messages
    labeled (int): Number of desired labeled images
    """
    if device is None and cuda_name is not None:
        device = torch.device(cuda_name)
    criterion = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0) 

    model = model.to(device)
    data_loader = load_data(dev_idx=-1, batch_size=128, dataset_name=dataset_name, seed=seed,
                            test_global=True, num_rooms=num_rooms, nyuv2_test=nyuv2_test)
    
    losses = AverageMeter()
    metrics = [AverageMeter() for _ in range(7)]

    model.eval()
    with torch.no_grad():
        for (images, depth) in data_loader:
            images = images.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            pred = model(images)

            loss = criterion(pred, depth)

            losses.update(loss.item(), images.size(0))

            errors = compute_errors(pred=pred, gt=depth, nyuv2_test=nyuv2_test)

            batch_size = images.size(0)
            for i in range(7):
                metrics[i].update(errors[i].item(), n=batch_size)

    # if verbose:
    #     print(f'[+] Test Loss: {losses.avg}, Test Deltas 1, 2, 3: {metrics[4].avg, metrics[5].avg, metrics[6].avg}')
        
    return losses.avg, metrics
                

def local_training(model_name, train_type, model_path, cuda_name, learning_rate, local_epochs, batch_size,
                   dev_idx, seed, N_fcl, coe, kt, buf_path="", verbose=False, comm_round=-1, num_rooms="3", data_iid=False):
    device = torch.device(cuda_name)    

    model = get_model(model_name=f"{model_name}", seed=seed, cuda_name=cuda_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(cuda_name)), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if verbose:
        print(f"[+] Device {dev_idx} training...")

    start_time = time.time()
    if train_type == "mde_fedcl":
        train_loss, train_metrics = train_fedcl(model=model, model_path=model_path, cuda_name=cuda_name,
                                                optimizer=optimizer, local_epochs=local_epochs, kt=kt,
                                                N_fcl=N_fcl, coe=coe, dev_idx=dev_idx,
                                                dataset_name=model_name.split('_')[0], seed=seed, comm_round=comm_round,
                                                batch_size=batch_size, num_rooms=num_rooms, data_iid=data_iid)

    elif train_type == "mde_fedknow":
        train_loss, train_metrics = train_fedknow(model=model, model_path=model_path, cuda_name=cuda_name,
                                                  optimizer=optimizer, local_epochs=local_epochs, kt=kt,
                                                  dev_idx=dev_idx, dataset_name=model_name.split('_')[0],
                                                  seed=seed,  batch_size=batch_size, num_rooms=num_rooms,
                                                  data_iid=data_iid)
    elif train_type == "mde_codeps":
        train_loss, train_metrics = (
            train_codeps(model_name=model_name, model_path=model_path, cuda_name=cuda_name, learning_rate=learning_rate,
                         local_epochs=local_epochs, kt=kt, dev_idx=dev_idx, dataset_name=model_name.split('_')[0],
                         seed=seed,batch_size=batch_size, buf_path=buf_path, num_rooms=num_rooms, data_iid=data_iid)
        )

    elif train_type == "mde_loca":
        train_loss, train_metrics = (
            train_loca(model=model, model_path=model_path, cuda_name=cuda_name, optimizer=optimizer, k1=local_epochs,
                       dev_idx=dev_idx, batch_size=batch_size, dataset_name=model_name.split('_')[0], kt=kt, seed=seed,
                       train_type=train_type, lr=learning_rate, num_rooms=num_rooms, data_iid=data_iid))
    
    train_time = time.time() - start_time
    return train_loss, train_metrics, train_time
