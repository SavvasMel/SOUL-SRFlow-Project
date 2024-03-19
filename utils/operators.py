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

import torch
import numpy as np
from utils.max_eigenval import max_eigenval

def blur_operators(kernel_len, size, type_blur, device, var = None):

    ch, nx, ny = size

    if type_blur=='uniform':
        h = torch.zeros(nx,ny)
        lx = kernel_len[0]
        ly = kernel_len[1]
        h[0:lx,0:ly] = 1/(lx*ly)
        c =  np.ceil((np.array([ly,lx])-1)/2).astype("int64")
    if type_blur=='gaussian' and var != None :
        h = torch.zeros(nx,ny)
        lx = kernel_len[0]
        ly = kernel_len[1] 
        [x,y] = torch.meshgrid(torch.ceil(torch.arange(-ly/2,ly/2)),torch.ceil(torch.arange(-lx/2,lx/2)), indexing ='xy')
        h[0:lx,0:ly] = torch.exp(-(x**2+y**2)/(2*var))
        h = h/torch.sum(h)
        c = np.ceil(np.array([lx,ly])/2).astype("int64") 

    H_FFT = torch.fft.fft2(torch.roll(h, shifts = (-c[0],-c[1]), dims=(0,1))).to(device)
    HC_FFT = torch.conj(H_FFT).to(device)
    
    # A forward operator
    A = lambda x: torch.real(torch.fft.ifft2(torch.multiply(H_FFT,torch.fft.fft2(x))))
    # A backward operator
    AT = lambda x: torch.real(torch.fft.ifft2(torch.multiply(HC_FFT,torch.fft.fft2(x))))

    AAT_norm = max_eigenval(A, AT, size, 1e-4, int(1e4), 0, device)

    return A, AT, AAT_norm