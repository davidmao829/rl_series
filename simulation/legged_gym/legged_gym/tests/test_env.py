# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, export_policy_as_jit, task_registry, Logger
from colorama import Fore, Style
import torch
import csv

def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.asset.fix_base_link = True
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.control.action_scale = 1
    env_cfg.init_state.pos = [0, 0 , 2.8]
    env_cfg.env.use_motor_model = True
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # init_csv('../Experiment/Actual_Swing_test.csv')
    init_csv('../Experiment/Ideal_torque.csv')

    for i in range(int(10 * env.max_episode_length)):
        # print(f"{Fore.GREEN}Ref_Dof_Pos{env.ref_dof_pos}{Style.RESET_ALL}")
        actions = env.ref_delta_action.clone()[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13]] #* 0
        # print(f"{Fore.RED}actions{actions}{Style.RESET_ALL}")
        #print(env.rigid_body_states[:, env.feet_indices, 2]) # 足部高度
        obs, _, rew, done, info = env.step(actions)
        actual_joint_pos = env.dof_pos.clone()[0]#obs[0, 10:24]
        target_joint_pos = env.ref_dof_pos.clone()[0]
        actual_torque = env.torques.clone()[0]
        ideal_torque = env.ideal_torque.clone()[0]
        print("actual_torque", actual_torque)
        print("ideal_torque", ideal_torque)
        torque = torch.cat([actual_torque, ideal_torque])
        joint_pos = torch.cat([actual_joint_pos, target_joint_pos])
        write_data_to_csv(torque, '../Experiment/Ideal_torque.csv')

    print("Done")

def init_csv(filename):
    """初始化CSV文件，清空其内容"""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as f:
        pass
    print(f"CSV file '{filename}' has been initialized.")

def write_data_to_csv(data, filename):
    """将单个时间步的关节位置写入CSV文件"""
    joint_pos = data  # 取出数据

    # 如果joint_pos是在GPU上，将其移到CPU并转换为普通Python列表
    if joint_pos.is_cuda:
        joint_pos = joint_pos.cpu().tolist()
    else:
        joint_pos = joint_pos.tolist()

    # 将数据写入CSV文件
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(joint_pos)

def write_ang_vel_to_csv(data, filename):
    """将单个时间步的关节位置写入CSV文件"""
    ang_vel = data  # 取出数据

    # 如果joint_pos是在GPU上，将其移到CPU并转换为普通Python列表
    if ang_vel.is_cuda:
        ang_vel = ang_vel.cpu().tolist()
    else:
        ang_vel = ang_vel.tolist()

    # 将数据写入CSV文件
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ang_vel)

if __name__ == '__main__':
    args = get_args()
    test_env(args)
