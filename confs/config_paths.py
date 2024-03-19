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

import argparse
import os
import datetime

def parse_paths(parent_parser):
    today = datetime.date.today()
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--conf_path", type=str, default='./confs/SRFlow_DF2K_8X.yml', help='Path of model settings')
    parser.add_argument("--lq_paths", type=str, default=os.path.join('./datasets/lr/256/8X', '*.png'), help='Path of LR images')
    parser.add_argument("--gt_paths", type=str, default=os.path.join('./datasets/gt/256', '*.png'), help="Path of HR/ground truth images")
    parser.add_argument("--path_results", type=str, default='./' + str(today) + '/SOUL_results', help='Path to save results')
    return parser