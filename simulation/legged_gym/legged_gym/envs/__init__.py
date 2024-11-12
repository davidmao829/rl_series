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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .base.humanoid import Humanoid

from .gr1.gr1_walk_phase import GR1WalkPhase
from .gr1.gr1_walk_phase_config import GR1WalkPhaseCfg, GR1WalkPhaseCfgPPO
from .gr1.gr1_5dof import GR1_5dof
from .gr1.gr1_5dof_config import GR1_5dofCfg, GR1_5dofCfgPPO

from .g1.g1_walk_phase import G1WalkPhase
from .g1.g1_walk_phase_config import G1WalkPhaseCfg, G1WalkPhaseCfgPPO
from .gr1_explicit.gr1_explicit import GR1_explicit
from .gr1_explicit.gr1_explicit_config import GR1_explicitCfg, GR1_explicitCfgPPO
from .h1.h1_walk_phase import H1WalkPhase
from .h1.h1_walk_phase_config import H1WalkPhaseCfg, H1WalkPhaseCfgPPO

from .berkeley.berkeley_walk_phase import BerkeleyWalkPhase
from .berkeley.berkeley_walk_phase_config import BerkeleyWalkPhaseCfg, BerkeleyWalkPhaseCfgPPO

from legged_gym.gym_utils.task_registry import task_registry

# ======================= environment registration =======================

task_registry.register("gr1_walk_phase", GR1WalkPhase, GR1WalkPhaseCfg(), GR1WalkPhaseCfgPPO())
task_registry.register("gr1_5dof", GR1_5dof, GR1_5dofCfg(), GR1_5dofCfgPPO())
task_registry.register("gr1_explicit", GR1_explicit, GR1_explicitCfg(), GR1_explicitCfgPPO())

task_registry.register("g1_walk_phase", G1WalkPhase, G1WalkPhaseCfg(), G1WalkPhaseCfgPPO())


task_registry.register("h1_walk_phase", H1WalkPhase, H1WalkPhaseCfg(), H1WalkPhaseCfgPPO())

task_registry.register("berkeley_walk_phase", BerkeleyWalkPhase, BerkeleyWalkPhaseCfg(), BerkeleyWalkPhaseCfgPPO())