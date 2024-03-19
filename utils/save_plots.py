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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.util import t, rgb

def save_plots(measure, true, post_meanvar_burnin, fixed_point, path):

    Image.fromarray(rgb(post_meanvar_burnin.get_mean())).save(path + "/postmean.png")
    Image.fromarray(rgb(fixed_point)).save(path + "/fixed_point.png")

    var_np = post_meanvar_burnin.get_var()[0].detach().cpu().numpy()
    var_grayscale = 0.299*var_np[0] + 0.587*var_np[1] + 0.114*var_np[2]
    norm_mean = matplotlib.colors.Normalize(vmin=np.min(np.sqrt(var_grayscale)), vmax=np.max(np.sqrt(var_grayscale)))

    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(var_grayscale), cmap = "gray", norm = norm_mean)
    plt.title("Posterior st. deviation")
    plt.colorbar(im)
    plt.savefig(path + "/poststdeviation.png", bbox_inches='tight')
    plt.close()

    res_np = (post_meanvar_burnin.get_mean()[0].cpu().numpy()-true[0].cpu().numpy())**2
    res_grayscale = 0.299*res_np[0] + 0.587*res_np[1] + 0.114*res_np[2]
    norm_mean = matplotlib.colors.Normalize(vmin=np.min(np.sqrt(res_grayscale)), vmax=np.max(np.sqrt(res_grayscale)))

    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(res_np.transpose([1,2,0])))
    plt.title("Residuals (squared-root)")
    plt.colorbar(im)
    plt.savefig(path + "/residual.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(res_grayscale), cmap = "gray", norm = norm_mean)
    plt.title("Residuals (squared-root, grayscaled)")
    plt.colorbar(im)
    plt.savefig(path + "/residual_gray.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(measure.PSNR_trace)
    plt.xlabel("$Iterations$")
    plt.ylabel("$PSNR$")
    plt.savefig(path + "/PSNR.png")
    plt.close()

    plt.figure()
    plt.plot(measure.SSIM_trace)
    plt.xlabel("$Iterations$")
    plt.ylabel("$SSIM$")
    plt.savefig(path + "/SSIM.png")
    plt.close()

    plt.figure()
    plt.plot(measure.LPIPS_trace)
    plt.xlabel("$Iterations$")
    plt.ylabel("$LPIPS$")
    plt.savefig(path + "/LPIPS.png")
    plt.close()

    plt.figure()
    plt.plot(measure.NRMSE_trace)
    plt.xlabel("$Iterations$")
    plt.ylabel("$NRMSE$")
    plt.savefig(path + "/NRMSE.png")
    plt.close()

    if type(np.hstack(measure.NRMSE_lr_trace)) == np.ndarray:
        plt.figure()
        plt.plot(measure.NRMSE_lr_trace)
        plt.xlabel("$Iterations$")
        plt.ylabel("$NRMSE (from true low-res.)$")
        plt.savefig(path + "/NRMSE(true).png")
        plt.close()
