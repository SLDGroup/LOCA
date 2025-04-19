from models.GuideDepth.model.loader import load_model


def get_model(model_name, seed, weights_pth=None, cuda_name="cuda:0", path="."):
    dataset_name, model_name = model_name.split('_')
    
    if model_name == "guidedepth":
        # MACs: 10.60 G
        # FLOPs: 5.82 M
        # Total params: 5,824,513
        # Input size (MB): 3.69
        # Forward/backward pass size (MB): 930.58
        # Params size (MB): 23.30
        # Estimated Total Size (MB): 957.56
        return load_model(model_name="GuideDepth", weights_pth=weights_pth, seed=seed, cuda_name=cuda_name, path=path)
    elif model_name == "guidedepth-s":
        # MACs: 6.10 G
        # FLOPs: 5.72 M
        # Total params: 5,723,273
        # Input size (MB): 3.69
        # Forward/backward pass size (MB): 426.77
        # Params size (MB): 22.89
        # Estimated Total Size (MB): 453.35
        return load_model(model_name="GuideDepth-S", weights_pth=weights_pth, seed=seed, cuda_name=cuda_name, path=path)
    else:
        raise NotImplementedError(f'Model {dataset_name}_{model_name} not implemented yet')
