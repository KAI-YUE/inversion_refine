
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names)
import lpips
import cv2
import nltk
import time
# from google.colab.patches import cv2_imshow
import torchvision
nltk.download('wordnet')
import logging
# from google.colab import files

from src.helper_functions.utils import *
from src.transformations.utils import *
from src.classification.utils import *
from src.optimization_functions.utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--image_path',type=str, default="dog.JPEG", required = True)

opt = parser.parse_args()

# get the image
ground_truth_image = adjust_image(opt.image_path)
# get the latent variable
latent_var = produce_truncated(1)
# dictionary
dict_of_losses = {
'lpips_alexnet' : lpips.LPIPS(net='alex'),
'L2': torch.nn.MSELoss(),
'L1': torch.nn.L1Loss(),
}

dict_of_sides={}

category = classify_image(ground_truth_image)
print(torch.where(category==1))
# category = torch.tensor([1])
# print(category)

# category = torch.zeros(1, 1000, dtype=torch.float32)
# category[0, 369] = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dict_of_dirs = get_image_transformations(ground_truth_image.to(device),1000,dict_of_losses,category=category)

# model = BigGAN.from_pretrained('biggan-deep-256').to(device)
# model.eval()

# for k, v in dict_of_dirs.items():
#     img = model(v)
#     pass

"""
python3 main.py --image_path dog.JPEG
"""