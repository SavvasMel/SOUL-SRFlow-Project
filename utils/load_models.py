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

# This file contains content licensed by 
# https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE
# https://github.com/andreas128/SRFlow/blob/master/LICENSE

import options.options as option
import os
import glob
import natsort

def get_resume_paths(opt):
    resume_state_path = None
    resume_model_path = None
    ts = opt_get(opt, ['path', 'training_state'])
    if opt.get('path', {}).get('resume_state', None) == "auto" and ts is not None:
        wildcard = os.path.join(ts, "*")
        paths = natsort.natsorted(glob.glob(wildcard))
        if len(paths) > 0:
            resume_state_path = paths[-1]
            resume_model_path = resume_state_path.replace('training_state', 'models').replace('.state', '_G.pth')
    else:
        resume_state_path = opt.get('path', {}).get('resume_state')
    return resume_state_path, resume_model_path

def opt_get(opt, keys, default=None):
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret

def load_model(conf_path, device):
    opt = option.parse(conf_path, is_train=False)
    if device=="cpu":
        opt['gpu_ids'] = None
    else:
        opt['gpu_ids'] = [int(device[-1])]
    opt = option.dict_to_nonedict(opt)
    if os.path.basename(conf_path.split('_')[0]) != "SRFlow":
        from models_DA import create_model
    else:
        from models import create_model
    model = create_model(opt)
    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt
