# Copyright (c) 2024 Savvas Melidonis, Matthew Holden, Yoann Altmann, Marcelo Pereyra, Konstantinos Zygalakis
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file contains content licensed by # https://github.com/andreas128/SRFlow/blob/master/LICENSE

#%% Load necessary packages

import torch
import numpy as np
import os
from PIL import Image

from argparse import ArgumentParser
from confs.config_paths import parse_paths
from utils.imresize import imresize
from utils.util import fiFindByWildcard, t, rgb
from utils.load_models import load_model
from utils.util import imread, set_random_seed
from utils.operators import blur_operators

from SOUL import SOUL

SEED = 666
set_random_seed(SEED)

parser = ArgumentParser()
parser.add_argument("--index_image", type = int, default = 1, help = "Index of image in the folder to be picked")
parser.add_argument("--type_blur", type = str, default = "uniform", help = "Choose blurring operator: uniform or gaussian")
parser.add_argument("--kernel_size", type = float, default = 9, help = "Kernel dimensions")
parser.add_argument("--kernel_std", type = float, default = 3, help = "Standard deviation for kernel")
parser.add_argument("--d_scale", type = float, default = 0.1, help = "Scale parameter for optim. stepsize")
parser.add_argument("--d_exp", type = float, default = 0.6, help = "Scale parameter for optim. stepsize. It should be in [0.6,0.9].")
parser.add_argument("--b_niter", type = int, default = 20, help = "Number of iterations for burn-in period")
parser.add_argument("--mn", type = int, default = 5, help = "Number of internal iterations for the sampling part of the algorithm")
parser.add_argument("--niter", type = int, default = 60000, help = "Number of iterations for the optimisation part of the algorithm")
parser.add_argument("--step_size", type = float, default = 5e-5, help = "Stepsize for the sampling part of the algorithm")
parser.add_argument("--sigma_z", type = float, default = 0.95, help = "Standard deviation on z (prior distribution)")

parser = parse_paths(parser)
hparam = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

#%% Load the model

model, opt = load_model(hparam.conf_path, device)

#%% Load datasets/images pathways, create folder to save results

os.makedirs(hparam.path_results, exist_ok=True)  

lq_paths = fiFindByWildcard(hparam.lq_paths) 
gt_paths = fiFindByWildcard(hparam.gt_paths) 

lqs = [imread(p) for p in lq_paths]
gts = [imread(p) for p in gt_paths]

# %% Load image and create synthetic data

gt = gts[hparam.index_image] 
gt_tensor = t(gt).to(device)
Image.fromarray(gt).save(hparam.path_results + "/groundtrue.png")

# lq = lqs[index] 
lq = imresize(gt,0.125)
lq_tensor = t(lq).to(device)
Image.fromarray(lq).save(hparam.path_results + "/lr_image.png")

kernel_len = [hparam.kernel_size, hparam.kernel_size]
size = list(gt_tensor[0].size())
A, AT, AAT_norm = blur_operators(kernel_len, size, hparam.type_blur, device , var = 3**2)
sigma = 5/255
y = A(gt_tensor) + sigma*torch.randn_like(gt_tensor)
Image.fromarray(rgb(y)).save(hparam.path_results + "/noisy.png")

#%% Define prior, likelihood and posterior distributions

def log_prior(z):

    normalising_constant = -(z.numel() / 2) * np.log(2 * np.pi * hparam.sigma_z**2)
    return normalising_constant - torch.sum(z**2,dim=(1,2,3)) / (2 * hparam.sigma_z**2)

def log_likelihood(x):
    
    normalising_constant = -(y.numel() / 2) * np.log(2 * np.pi * sigma**2)
    return normalising_constant - torch.sum((y-A(x))**2 / (2 * sigma**2),dim=(1,2,3))

def log_posterior(x, z):

    return  log_prior(z) + log_likelihood(x)# log_likelihood(forw_conv(A,x))

#%% Initializations

# Define initialization for z
z= model.get_encode_z(lq_tensor, gt_tensor)
z = hparam.sigma_z*torch.randn_like(z)

# Define initialization for the low-resolution image
lq_init = t(imresize(rgb(y),0.125)).to(device)
Image.fromarray(rgb(lq_init)).save(hparam.path_results + "/initoflr.png")

print(' ')
print('Run SOUL-SRFLOW...')
SOUL(gt, lq, z, lq_init, model, log_likelihood, log_prior, log_posterior, hparam, hparam.path_results)
print(' ')