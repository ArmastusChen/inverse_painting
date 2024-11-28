import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class LPIPS_Image_Processor():
    def __init__(self):
        print("Loading LPIPS model")

    def process(self, img):
        # input PIL
        img = T.ToTensor()(img).unsqueeze(0)

        # Normalize to [-1, 1]
        norm_img = 2 * img - 1

        return img, norm_img


class Seg_Image_Processor():
    def __init__(self):
        super(Seg_Image_Processor, self).__init__()



    def process(self, img):
        # input PIL
        img = T.ToTensor()(img).unsqueeze(0)

        # Normalize to [-1, 1]
        # norm_img = 2 * img - 1

        return img
    


        