import torch
import _pickle as cpickle
from utils.dataset_utils import load_data
from utils.general_utils import AverageMeter
from utils.loss_utils import Depth_Loss
from utils.codeps.codeps_utils import EdgeAwareSmoothnessLoss, DiversityBuffer
from utils.testing_utils import compute_errors
from models.get_model import get_model
from os import path
import random


def train_codeps(model_name, dev_idx, model_path, cuda_name, learning_rate, buf_path, local_epochs, dataset_name,
                 seed, kt, batch_size=20, num_rooms="3", data_iid=False):
    device = torch.device(cuda_name)
    # Putting strong=True makes the model return y, y_features
    model = get_model(model_name=f"{model_name}", seed=seed, cuda_name=cuda_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(cuda_name)), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_loaders, test_loaders = load_data(test_global=False, dev_idx=dev_idx, dataset_name=dataset_name, seed=seed,
                                            batch_size=batch_size, train_type="fedcl", train_split=0.8,
                                            num_rooms=num_rooms, data_iid=data_iid)

    # if dev_idx == 1:
    #     print(f"Pretraining on cuda {cuda_name}")
    depth_loss = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)
    smth_loss = EdgeAwareSmoothnessLoss()
    train_loss = AverageMeter()
    deltas = [AverageMeter() for _ in range(3)]
    model.train()
    for epoch_idx in range(local_epochs):
        for loader_idx in range(len(train_loaders)):
            if loader_idx == 0:
                buffer = DiversityBuffer(feature_dim=32, max_size=int(num_rooms[-1]) * 25, similarity_threshold=0.80)
            else:
                with open(buf_path, 'rb') as f:
                    buffer = cpickle.load(f)
            loader = train_loaders[loader_idx]
            for kt_idx in range(kt):
                for batch_idx, batch in enumerate(loader):
                    _, _, image, depth = batch
                    if image.size(0) == 1:
                        continue
                    # if dev_idx == 1:
                    #     print(f"{dev_idx}: Training starting now")
                    image = image.to(device, non_blocking=True)
                    depth = depth.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    # if dev_idx == 1:
                    #     print(f"{dev_idx}: Forward pass")
                    pred, features = model(image, get_features=True)
                    
                    if len(buffer) > batch_size:
                        buffer_features, buffer_imgs, feature_depths = buffer.sample_features(batch_size=batch_size)

                        buffer_features = buffer_features.to(device, non_blocking=True)
                        buffer_imgs = buffer_imgs.to(device, non_blocking=True)
                        feature_depths = feature_depths.to(device, non_blocking=True)
                        
                        pred_features = model.decoder(buffer_imgs, buffer_features)
                        
                        pred_total = torch.cat((pred, pred_features), dim=0)
                        depth_total = torch.cat((depth, feature_depths), dim=0)
                        loss = depth_loss(pred_total, depth_total) + 0.001 * smth_loss(pred_total, depth_total)
                    else:
                        loss = depth_loss(pred, depth) + 0.001 * smth_loss(pred, depth)
                    
                    # Need to analyze every single frame (according to literature)
                    for i in range(len(image)):
                        # print("FRAME:", i)
                        buffer.add_feature(features[i], image[i], depth[i], random.randint(0, 255))

                    train_loss.update(loss.item(), image.size(0))
                    loss.backward()
                    optimizer.step()
            # print(f"Device {dev_idx}, loss: {train_loss.avg}")
            
            with open(buf_path, 'wb') as f:
                cpickle.dump(buffer, f)
    torch.save(model.state_dict(), model_path)
    return train_loss.avg, deltas