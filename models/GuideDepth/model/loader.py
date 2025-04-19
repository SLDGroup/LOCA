import os.path
import torch
from models.GuideDepth.model.GuideDepth import GuideDepth


def load_model(model_name, weights_pth, seed, cuda_name="cuda:0", path="."):
    model = model_builder(model_name=model_name, seed=seed, path=path, cuda_name=cuda_name)

    if weights_pth is not None:
        ckpt = torch.load(weights_pth, map_location=cuda_name)
        model.load_state_dict(ckpt)

    return model


def model_builder(model_name, seed, path=".", cuda_name="cuda:0"):
    if model_name == 'GuideDepth':
        return GuideDepth(pretrained=True, seed=seed, path=path, cuda_name=cuda_name)
    if model_name == 'GuideDepth-S':
        return GuideDepth(pretrained=True, up_features=[32, 8, 4], inner_features=[32, 8, 4], seed=seed, path=path, cuda_name=cuda_name)

    print("Invalid model")
    exit(0)


