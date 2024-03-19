# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file contains content licensed by https://github.com/andreas128/SRFlow/blob/master/LICENSE

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

class metrics():
    
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net, verbose=False)
        self.model.to(self.device)
        self.PSNR_trace = []
        self.SSIM_trace = []
        self.LPIPS_trace = []
        self.NRMSE_trace = []
        self.NRMSE_lr_trace = []
        
    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips, self.nrmse]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def nrmse(self, imgA, imgB):
        return np.linalg.norm(((imgA/255.)-(imgB/255.)).ravel(),2)/np.linalg.norm((imgB/255.).ravel(),2)

    def ssim(self, imgA, imgB):
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        score, diff = ssim(imgA, imgB, full=True, multichannel=True)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val

    def update(self, x, t):
        measures = self.measure(x, t)
        self.PSNR_trace.append(measures[0])
        self.SSIM_trace.append(measures[1])
        self.LPIPS_trace.append(measures[2])
        self.NRMSE_trace.append(measures[3])

    def update_lr(self, x, lr):  
        self.NRMSE_lr_trace.append(self.nrmse(x, lr))

    def return_metrics(self):
        return self.PSNR_trace[-1], self.SSIM_trace[-1], self.LPIPS_trace[-1], self.NRMSE_trace[-1]

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1