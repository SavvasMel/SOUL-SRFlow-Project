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

import joblib
import numpy as np

def save_progress(index, current_z, fixed_point_z, lq_last, estimate, measure, post_meanvar, chain, path):

    data = {
    "index": index,
    "last_sample_z": current_z.detach().cpu().numpy(),
    "fixed_point_z": fixed_point_z.detach().cpu().numpy(),
    "lq_last": lq_last.detach().cpu().numpy(),
    "estimate": estimate.cpu().numpy(),
    "PSNR_trace": np.stack(measure.PSNR_trace),
    "SSIM_trace": np.stack(measure.SSIM_trace),
    "LPIPS_trace" : np.stack(measure.LPIPS_trace),
    "NRMSE_trace" : np.stack(measure.NRMSE_trace),
    "NRMSE_lr_trace": np.stack(measure.NRMSE_lr_trace),
    "meanSamples": post_meanvar.get_mean().cpu().numpy(),
    "variance": post_meanvar.get_var().cpu().numpy()}

    with open(path + '/data_SOUL.joblib', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        joblib.dump(data, f)

    with open(path + '/MC_chain.joblib', 'wb') as f:
        joblib.dump({'MC_chain' : np.stack(chain)}, f)