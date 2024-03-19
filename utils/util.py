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

# Parts of this repository are licensed by
# https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE
# https://github.com/andreas128/SRFlow/blob/master/LICENSE

import glob
import random
from collections import OrderedDict
import natsort
import numpy as np
import torch
import cv2
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0)) / 255. #.astype(np.float32)) / 255

def rgb(t): return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)

def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]

class welford: 
    def __init__(self, x):
        self.k = 1
        self.M = x
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M)*(x - Mnext)
        self.M = Mnext
    
    def get_mean(self):
        return self.M
    
    def get_var(self):
        return self.S/(self.k-1)