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

import numpy as np
import torch
from utils.util import welford
from utils.save_progress import save_progress
from utils.save_plots import save_plots
from utils.util import t, rgb
import time as time
from tqdm.auto import tqdm
from PIL import Image
from utils import metrics

def SOUL(gt, lqt, z, mu_n, model, log_likelihood, log_prior, log_posterior, hparam, path):

    print('Iterations: {}'.format(hparam.b_niter+hparam.niter*hparam.mn))
    print('mn: {}'.format(hparam.mn))
    print('Stepsize: {}'.format(hparam.step_size))
    print('st.dev. of z: {}'.format(hparam.sigma_z))
    print('d_scale: {:.4f}'.format(hparam.d_scale))
    print('d_exp: {:.4f}'.format(hparam.d_exp))
    print(' ')

    # Write results in a text file
    text_file = open(hparam.path_results + '/results.txt', "w+")
    text_file.write('Launch SOUL-SRFLOW!')
    text_file.write('Iterations: {}\n'.format(hparam.b_niter+hparam.niter*hparam.mn))
    text_file.write('mn: {}\n'.format(hparam.mn))
    text_file.write('Stepsize: {}\n'.format(hparam.step_size))
    text_file.write('st.dev. of z: {}\n'.format(hparam.sigma_z))
    text_file.write('d_scale: {:.4f}\n'.format(hparam.d_scale))
    text_file.write('d_exp: {:.4f}\n'.format(hparam.d_exp))
    text_file.close()

    # Load the metrics to measure performance 

    if torch.cuda.is_available():
        measure = metrics.metrics(use_gpu=True)
    else:
        measure = metrics.metrics(use_gpu=False)
    
    # Save samples from the MC chain (thinning)
        
    MC_z = []
    save_samples = 2000
    thinning_step = np.int64(hparam.niter * hparam.mn/save_samples)

    # Initialize

    sr_new = model.get_sr(lq = mu_n, z = z.requires_grad_(), heat = 0)
    post_meanvar_burnin = welford(sr_new.detach())
    k = 0

    print (" Burn-in phase started!")

    start_time = time.time()

    for i in tqdm(range(0, hparam.b_niter)):

        U = -log_posterior(sr_new, z)
        dUdz = torch.autograd.grad(U, z)[0]
        z = z.detach() - hparam.step_size * dUdz + np.sqrt(2 * hparam.step_size) * torch.randn_like(z.detach())
        sr_new = model.get_sr(lq=mu_n, z=z.requires_grad_(), heat = 0)
        
        post_meanvar_burnin.update(sr_new.detach())
        measure.update(rgb(post_meanvar_burnin.get_mean()), gt)

        k = k+1
        text_file = open(path + '/results.txt', "a+")
        text_file.write('''Iteration (burn-in) [{}/{}], PSNR: {:.2f}, SSIM : {:.2f}, LPIPS : {:.4f}, NRMSE : {:.4f} \n'''
                        .format(k+1, int(hparam.b_niter + hparam.niter * hparam.mn), *measure.return_metrics()))
        text_file.close()

    print (" Burn-in phase finished! ")

    # delta(i) steps for SOUL algorithm 
    delta = lambda l: hparam.d_scale*( (l**(-hparam.d_exp)) / z.numel() )

    weighted_mu_n = torch.zeros_like(mu_n)
    delta_sum = 0
    count = 0

    U = -log_posterior(sr_new, z)
    dUdz = torch.autograd.grad(U, z)[0]
    max_U = -U.detach().clone()
    post_meanvar = welford(sr_new.detach().clone())

    for i in tqdm(range(0, hparam.niter)):
        
        H = torch.zeros_like(mu_n)

        for j in range(0, hparam.mn):
            
            z = z.detach() - hparam.step_size * dUdz + np.sqrt(2 * hparam.step_size) * torch.randn_like(z.detach())
            sr_new = model.get_sr(lq=mu_n.requires_grad_(), z=z.requires_grad_(), heat = 0)
            U = -log_likelihood(sr_new)-log_prior(z)
            if max_U < -U:
                max_point = sr_new.detach().clone()
                max_point_z = z.detach().clone()
                max_U = -U.detach().clone()
            dUdz, dFdlq = torch.autograd.grad(U, (z,mu_n))
            H += (1/hparam.mn)*(-dFdlq)
            
            post_meanvar_burnin.update(sr_new.detach())
            post_meanvar.update(sr_new.detach())
            measure.update(rgb(post_meanvar_burnin.get_mean()), gt)

            if count == thinning_step-1:
                MC_z.append(z.detach().cpu().numpy())
                count = 0
            else:
                count += 1     
            k = k+1

        # Optimisation step
            
        mu_n = torch.clamp(mu_n.detach() + delta(i+1)*H, 0, 1)
        delta_sum += delta(i+1)
        weighted_mu_n += delta(i+1)*mu_n.detach()
        mu_hat = weighted_mu_n/delta_sum
        measure.update_lr(rgb(mu_hat), lqt)

        if (i+1)%2 == 0:

            end_time = time.time()
            elapsed = end_time - start_time

            text_file = open(path + '/results.txt', "a+")
            text_file.write('''Iteration (burn-in) [{}/{}], PSNR: {:.3f}, SSIM : {:.3f}, LPIPS : {:.4f}, NRMSE : {:.4f}, elapsed time: {:.5f}
                            \n'''.format(k, int(hparam.b_niter + hparam.niter * hparam.mn), *measure.return_metrics(), elapsed))
            text_file.close()

            print("************************************************")
            print("Metrics of lr estimate ---> PSNR: {:.3f}".format(measure.psnr(rgb(mu_hat), lqt)))
            print(" ")
            print("Metrics posterior mean ---> PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.4f}, NRMSE: {:.4f}".format(*measure.return_metrics()))
            print(" ")
            print("Metrics of fixed point ---> PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.4f}, NRMSE: {:.4f}".format(*measure.measure(rgb(max_point), gt)))

        if (i+1)%10 == 0:
            save_plots(measure, t(gt), post_meanvar_burnin, max_point, path)

        if (i+1)%100 == 0:
            save_progress(i, z, max_point_z, mu_n, mu_hat, measure, post_meanvar, MC_z, path)

    return 