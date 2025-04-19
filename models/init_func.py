import torch.nn as nn
import torch.nn.init as init
from utils.general_utils import seed_everything

# Recursive function to apply initialization while checking for activation layers
def apply_initialization(model, seed):
    seed_everything(seed=seed)
    # Convert the model to a list of layers if it's a Sequential block
    modules = list(model.children())

    # Go through the layers and initialize them
    for i, layer in enumerate(modules):
        next_layer = modules[i + 1] if i + 1 < len(modules) else None
        init_weights(layer, next_layer)

        # If the current layer is a container like Sequential, apply initialization recursively
        if isinstance(layer, nn.Sequential):
            apply_initialization(layer, seed)

def init_weights(m, next_layer=None):
    # Initialize Conv2d layers
    if isinstance(m, nn.Conv2d):
        # print(f'Applying Kaiming (He) Initialization to Conv2d: {m}')
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    # Initialize Linear layers, with Xavier for Sigmoid, Kaiming for ReLU or unspecified
    elif isinstance(m, nn.Linear):
        if isinstance(next_layer, nn.Sigmoid):
            # print(f'Applying Xavier Initialization to Linear (Sigmoid Activation Follows): {m}')
            init.xavier_normal_(m.weight)
        else:
            # print(f'Applying Kaiming (He) Initialization to Linear (ReLU or unspecified): {m}')
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        if m.bias is not None:
            init.constant_(m.bias, 0)

    # Initialize BatchNorm2d layers
    elif isinstance(m, nn.BatchNorm2d):
        # print(f'Applying constant weight and bias initialization to BatchNorm2d: {m}')
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)