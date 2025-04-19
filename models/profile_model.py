import torch
from thop import profile
from torchinfo import summary

def profile_model_cpu(model, input_shape=(1, 3, 480, 640)):
    # Create a dummy input tensor with the appropriate shape
    # Assuming the input to the model is 1D, adjust shape as per your input size
    dummy_input = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3])  # Batch size 1, input size 64

    # Use `thop` to calculate FLOPs and MACs
    macs, flops = profile(model, inputs=(dummy_input,))
    print("Model Summary (Layer-wise details):")
    summary(model, input_size=(1, 3, 480, 640))

    # Convert the number of MACs/FLOPs to human-readable format
    def convert_size(num):
        for unit in [' ', 'K', 'M', 'G', 'T']:
            if num < 1000.0:
                return f"{num:.2f} {unit}"
            num /= 1000.0
        return f"{num:.2f} P"


    # Print the MACs, FLOPs, and parameter details
    print(f'\nMACs: {convert_size(macs)}')
    print(f'FLOPs: {convert_size(flops)}')