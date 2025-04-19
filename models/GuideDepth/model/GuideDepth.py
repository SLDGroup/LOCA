import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GuideDepth.model.DDRNet_23_slim import DualResNet_Backbone
from models.GuideDepth.model.modules import Guided_Upsampling_Block, SELayer
from models.init_func import apply_initialization

class Decoder(nn.Module):
    def __init__(self, up_1, up_2, up_3):
        super(Decoder, self).__init__()
        self.up_1 = up_1
        self.up_2 = up_2
        self.up_3 = up_3

    def forward(self, x, y_enc):
        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)
        
        # Assuming y_enc is the encoder output you want to pass through the decoder
        y = F.interpolate(y_enc, scale_factor=2, mode='bilinear')
        y = self.up_1(x_quarter, y)
       
        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_2(x_half, y)
       
        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_3(x, y)
        
        return y

class GuideDepth(nn.Module):
    def __init__(self,
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16],
            seed=42,
            path=".",
            cuda_name="cuda:0"):
        super(GuideDepth, self).__init__()
        self.feature_extractor = DualResNet_Backbone(pretrained=pretrained, features=up_features[0], seed=seed,
                                                     path=path, cuda_name=cuda_name)

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full", seed=seed)
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full", seed=seed)
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full", seed=seed)
        
        self.decoder = Decoder(up_1=self.up_1, up_2=self.up_2, up_3=self.up_3)
        
        apply_initialization(model=self, seed=seed)

    def forward(self, x, get_features=False):
        y_enc = self.feature_extractor(x)
        y = self.decoder(x, y_enc)

        if get_features:
            return y, y_enc

        return y
