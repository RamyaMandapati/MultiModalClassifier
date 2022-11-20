import torch
import torch_tensorrt
import timm
import time
import numpy as np
import torch.backends.cudnn as cudnn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

efficientnet_b0 = timm.create_model('efficientnet_b0',pretrained=True)

print("Hello world")
