from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, POSE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.humanoid import Humanoid
# from .humanoid_config import HumanoidCfg
from .gr1_walk_phase_config import GR1WalkPhaseCfg
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.gym_utils.math import *
from legged_gym.utils.util_math import *
from legged_gym.gym_utils.motor_delay_fft import MotorDelay_130, MotorDelay_80
from scipy.interpolate import CubicSpline


class GR1_wb(Humanoid):
    def __init__(self, cfg: GR1WalkPhaseCfg, sim_params, physics_engine, sim_device, headless):
        self.control_index = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13])
        self.cfg = cfg
        self.use_motor_model = self.cfg.env.use_motor_model
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        phase = self._get_phase()
        self.compute_ref_state()

    # def  _get_phase(self):
    #     cycle_time = self.cfg.rewards.cycle_time
    #     if self.cfg.commands.sw_switch:
    #         stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #         self.phase_length_buf[stand_command] = 0
    #         # self.gait_start is rand 0 or 0.5
    #         phase = (self.phase_length_buf * self.dt / cycle_time) * (~stand_command)
    #     else:
    #         phase = self.episode_length_buf * self.dt / cycle_time
    #     # phase continue increase, if want robot stand, set 0
    #
    #     return phase

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def generate_gait_time(self, envs):
        if len(envs) == 0:
            return

        # rand sample
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                                    gait_time_range[1],
                                                    (len(envs), 1), device=self.device)
            random_tensor_list.append(random_tensor_single)
        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        current_sum = torch.sum(random_tensor, dim=1, keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:, 1:] = scaled_tensor[:, :-1].clone()
        scaled_tensor[:, 0] *= 0.0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait2__|
        self.gait_time[envs] = torch.cumsum(scaled_tensor, dim=1).int()

    def _resample_commands(self):
        """
        Randomly select commands of some environments
        """

        for i in range(len(self.cfg.commands.gait)):
            env_ids = (self.episode_length_buf == self.gait_time[:, i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                resample_command(env_ids)

    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                     self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)

    # def compute_ref_state(self):
    #     phase = self._get_phase()
    #     _sin_pos_l = torch.sin(2 * torch.pi * phase)
    #     _sin_pos_r = torch.sin(2 * torch.pi * phase + torch.pi)
    #     sin_pos_l = _sin_pos_l.clone()
    #     sin_pos_r = _sin_pos_r.clone()
    #     self.ref_dof_pos = torch.zeros_like(self.dof_pos)
    #     scale_1 = self.cfg.rewards.target_joint_pos_scale
    #     scale_2 = 2 * scale_1
    #     # left foot stance phase set to default joint pos
    #     sin_pos_l[sin_pos_l > 0] = 0
    #     ratio_l = torch.clamp(torch.abs(sin_pos_l) - self.cfg.rewards.double_support_threshold, min=0, max=1) / \
    #                 (1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_l)
    #     self.ref_dof_pos[:, 2] = ratio_l * scale_1
    #     self.ref_dof_pos[:, 3] = -ratio_l * scale_2
    #     self.ref_dof_pos[:, 4] = ratio_l * scale_1
    #     # right foot stance phase set to default joint pos
    #     sin_pos_r[sin_pos_r > 0] = 0
    #     ratio_r = torch.clamp(torch.abs(sin_pos_r) - self.cfg.rewards.double_support_threshold, min=0, max=1) / (
    #                 1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_r)
    #     self.ref_dof_pos[:, 8] = ratio_r * scale_1
    #     self.ref_dof_pos[:, 9] = -ratio_r * scale_2
    #     self.ref_dof_pos[:, 10] = ratio_r * scale_1
    #     # Double support phase
    #     indices = (torch.abs(sin_pos_l) < self.cfg.rewards.double_support_threshold) & (
    #                 torch.abs(sin_pos_r) < self.cfg.rewards.double_support_threshold)
    #     self.ref_dof_pos[indices] = 0
    #     self.ref_dof_pos += self.default_dof_pos_all
    #     # print("ref_dof_pos: ", self.ref_dof_pos[0, [2,3,4,8,9,10]])
    #     self.ref_action = 2 * self.ref_dof_pos
    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time  # cycle_time = 0.8s
        phase = self.episode_length_buf * self.dt % cycle_time
        return phase

    def _get_gait_phase(self):  # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        cycle_time = self.cfg.rewards.cycle_time  # cycle_time = 0.8s
        stance_mask = torch.zeros((self.num_envs, 2),
                                  device=self.device)  # Left foot stance (0.0 ~ 0.04s and 0.76 ~ 0.8s)
        stance_mask[:, 0] = (phase >= (0.5 * cycle_time))  # Left foot stance (0.4 ~ 0.8s)
        stance_mask[:, 1] = (phase < (0.5 * cycle_time))  # Left foot stance (0.0 ~ 0.4s)
        stance_mask[(phase < (0.05 * cycle_time)) | (
                phase >= (0.95 * cycle_time))] = 1  # Double support phase (0.0 ~ 0.02s and 0.78 ~ 0.8s)
        self.ref_mask = 1 - stance_mask
        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()  # phase shape: [4096]
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        # phase = phase % 1.0  # Normalize phase to [0, 1)# Initialize reference positions
        self.ref_dof_pos[:, :] = self.default_dof_pos[0, :]  # shape: [4096, 12]# Define the fixed pose (双支撑姿态)
        cycle_time = self.cfg.rewards.cycle_time  # cycle_time = 0.8s
        fixed_pose = torch.tensor([0.0, 0.0, -0.1, 0.2, -0.1, 0.0],
                                  device=self.device)  # shape: [6]# Define the stepping trajectory (左腿摆动和右腿摆动的轨迹)
        stepping_traj = [[0.0, 0.0, -0.1, 0.2, -0.1, 0.0, ],  # t = 0.0s
                         [0.0, 0.0, -0.26, 0.4, -0.14, 0.0],  # t = 0.1s
                         [0.0, 0.0, -0.4, 0.64, -0.24, 0.0],  # t = 0.2s
                         [0.0, 0.0, -0.26, 0.4, -0.14, 0.0],  # t = 0.3s
                         [0.0, 0.0, -0.1, 0.2, -0.1, 0.0]  # t = 0.4s，与 t=0.0s相同
                         ]
        arm_traj = [[0, 0],
                    [0.3, -0.3],
                    [0, 0],
                    [-0.3, 0.3],
                    [0, 0]]
        arm_traj = torch.tensor(arm_traj, device=self.device)

        stepping_traj = torch.tensor(stepping_traj,
                                     device=self.device)  # shape: [5, 6]# Define the time points for the stepping trajectory
        offline_times = torch.tensor([0, 0.125, 0.25, 0.375, 0.5], device=self.device) * cycle_time
        arm_offline_times = torch.tensor([0, 0.25, 0.5, 0.75, 1.0], device=self.device) * cycle_time
        stepping_traj_np = stepping_traj.cpu().numpy()  # shape: [5, 6]
        arm_traj_np = arm_traj.cpu().numpy()  # shape: [5, 6]
        offline_times_np = offline_times.cpu().numpy()  # shape: [5]# Interpolate the stepping trajectory using cubic spline
        arm_offline_times_np = arm_offline_times.cpu().numpy()  # shape: [5]# Interpolate the stepping trajectory using cubic spline
        cs = CubicSpline(offline_times_np, stepping_traj_np, axis=0)
        arm_cs = CubicSpline(arm_offline_times_np, arm_traj_np, axis=0)

        # left_stand_mask= (phase <=0.5*cycle_time)&(phase>=0.0)
        # right_stand_mask= (phase >0.5*cycle_time) & (phase<=cycle_time)
        mask_double_support_1 = (phase < (0.05 * cycle_time))  # 双支撑阶段：0.0 ~ 0.04s
        mask_left_swing = (phase >= (0.05 * cycle_time)) & (phase < (0.5 * cycle_time))  # 左腿摆动阶段：0.04 ~ 0.4s
        # print(f"{Fore.GREEN}mask_left_swing: {mask_left_swing}{Style.RESET_ALL}")
        mask_double_support_2 = (phase >= (0.45 * cycle_time)) & (phase <= (0.55 * cycle_time))
        mask_right_swing = (phase >= (0.5 * cycle_time)) & (phase < (0.95 * cycle_time))  # 右腿摆动阶段：0.4 ~ 0.76s
        # print(f"{Fore.LIGHTCYAN_EX}mask_right_swing: {mask_right_swing}{Style.RESET_ALL}")
        mask_double_support_3 = (phase >= (
                0.95 * cycle_time))  # 双支撑阶段：0.76 ~ 0.8s# Compute t_virtual for left and right swing phases
        t_virtual_left = ((phase[mask_left_swing] - (0.05 * cycle_time)) / (0.45 * cycle_time)) * (0.5 * cycle_time)
        t_virtual_right = ((phase[mask_right_swing] - (0.55 * cycle_time)) / (0.45 * cycle_time)) * (
                0.5 * cycle_time)  # Interpolate for left and right swing phases
        arm_np = arm_cs(np.array(phase.tolist())).squeeze()
        self.ref_dof_pos[:, [12, 13]] = torch.tensor(arm_np, device=self.device,
                                                     dtype=torch.float32)  # Convert to Float

        # Interpolate for left and right swing phases
        if mask_left_swing.any():
            left_command_np = cs(t_virtual_left.cpu().numpy())  # Interpolate using numpy
            left_command = torch.tensor(left_command_np, device=self.device, dtype=torch.float32)  # Convert to Float
            self.ref_dof_pos[mask_left_swing, 0:6] = left_command
            self.ref_dof_pos[mask_left_swing, 6:12] = fixed_pose.float()  # 右腿保持固定姿态
        if mask_right_swing.any():
            right_command_np = cs(t_virtual_right.cpu().numpy())  # Interpolate using numpy
            right_command = torch.tensor(right_command_np, device=self.device, dtype=torch.float32)  # Convert to Float
            self.ref_dof_pos[mask_right_swing, 6:12] = right_command
            self.ref_dof_pos[mask_right_swing, 0:6] = fixed_pose.float()  # 左腿保持固定姿态

        self.ref_dof_pos[mask_double_support_1 | mask_double_support_2 | mask_double_support_3, 0:6] = fixed_pose
        self.ref_dof_pos[mask_double_support_1 | mask_double_support_2 | mask_double_support_3, 7:13] = fixed_pose
        self.ref_action = 2 * self.ref_dof_pos

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _init_buffers(self):
        super()._init_buffers()
        if self.use_motor_model:
            self.motordelay0 = MotorDelay_80(self.num_envs, 1, device=self.device)
            self.motordelay6 = MotorDelay_80(self.num_envs, 1, device=self.device)

            self.motordelay2 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay3 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay8 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay9 = MotorDelay_130(self.num_envs, 1, device=self.device)

            self.fric_para_0 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_1 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_2 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_3 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric = torch.zeros(self.num_envs, self.num_dofs,
                                    dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait), dtype=torch.int, device=self.device,
                                     requires_grad=False)

    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        dof_pos = self.default_dof_pos_all.clone()

        # reset robot states
        self._reset_dofs(env_ids, dof_pos, torch.zeros_like(dof_pos))
        self._reset_root_states(env_ids)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.feet_land_time[env_ids] = 0.
        self.phase_length_buf[env_ids] = 0
        self._reset_buffers_extra(env_ids)
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

        # resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids] * self.reward_scales[key]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        count = torch.sum(torch.norm(self.commands[:, :3], dim=1) < self.cfg.commands.stand_com_threshold).item()
        self.extras["episode"]["count_stand"] = count
        return

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        self.phase_length_buf += 1
        self._resample_commands()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _get_body_indices(self):
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                         requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                          torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                             requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                              upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                             requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                              lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         knee_names[i])

    def _reset_buffers_extra(self, env_ids):
        if self.use_motor_model:
            self._reset_fric_para(env_ids)
            self.motordelay0.reset(env_ids)
            self.motordelay6.reset(env_ids)
            self.motordelay2.reset(env_ids)
            self.motordelay3.reset(env_ids)
            self.motordelay8.reset(env_ids)
            self.motordelay9.reset(env_ids)

    def _reset_fric_para(self, env_ids):
        self.fric_para_0[env_ids, 0] = torch_rand_float(3.7, 6.6, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 1] = torch_rand_float(3.3, 5.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 2] = torch_rand_float(-5.0, -3.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 3] = torch_rand_float(0.7, 0.9, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 4] = torch_rand_float(0.7, 0.9, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_1[env_ids, 0] = torch_rand_float(1.2, 2.75, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 1] = torch_rand_float(1.0, 1.55, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 2] = torch_rand_float(-1.55, -1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 3] = torch_rand_float(0.4, 0.65, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 4] = torch_rand_float(0.4, 0.65, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_2[env_ids, 0] = torch_rand_float(1.9, 3.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 1] = torch_rand_float(1.15, 2.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 2] = torch_rand_float(-2.0, -1.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 3] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 4] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_3[env_ids, 0] = torch_rand_float(0.25, 1.25, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 1] = torch_rand_float(0.2, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 2] = torch_rand_float(-1.0, -0.2, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 3] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 4] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)

    def _compute_torques(self, actions):
        # pd controller
        actions_scaled_raw = actions * self.cfg.control.action_scale
        actions_scaled = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        actions_scaled[:, self.control_index] = actions_scaled_raw
        # print("ref_pos", self.ref_dof_pos)
        # print("actions_scaled_raw", actions_scaled_raw)
        # print("control_index", self.control_index)
        # print("actions_scaled", actions_scaled)
        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_motor:
                torques = self.p_gains * (
                        actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains * self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains * (
                        actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[
                              1] * self.d_gains * self.dof_vel

        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.use_motor_model:
            torques[:, [0]] = self.motordelay0(torques[:, [0]])
            torques[:, [6]] = self.motordelay6(torques[:, [6]])
            torques[:, [2]] = self.motordelay2(torques[:, [2]])
            torques[:, [3]] = self.motordelay3(torques[:, [3]])
            torques[:, [8]] = self.motordelay8(torques[:, [8]])
            torques[:, [9]] = self.motordelay9(torques[:, [9]])

            self.friction_0_6()
            self.friction_1_7()
            self.friction_2_3_8_9()

            torques -= self.fric

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def friction_0_6(self):
        flag_0 = (self.dof_vel[:, 0] <= 0.002) & (self.dof_vel[:, 0] >= -0.002)
        flag_1 = ((self.dof_vel[:, 0] > 0.002) & (self.dof_vel[:, 0] <= 0.16))
        flag_2 = (self.dof_vel[:, 0] > 0.16)
        flag_3 = ((self.dof_vel[:, 0] < -0.002) & (self.dof_vel[:, 0] >= -0.16))
        flag_4 = (self.dof_vel[:, 0] < -0.16)

        self.fric[:, 0] = self.fric_para_0[:, 0] / 0.002 * self.dof_vel[:, 0] * flag_0 + \
                          ((self.fric_para_0[:, 1] - self.fric_para_0[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 0] - 0.002) + self.fric_para_0[:, 0]) * flag_1 + \
                          (self.fric_para_0[:, 1] + self.fric_para_0[:, 3] * (self.dof_vel[:, 0] - 0.16)) * flag_2 + \
                          ((self.fric_para_0[:, 2] + self.fric_para_0[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 0] + 0.002) - self.fric_para_0[:, 0]) * flag_3 + \
                          (self.fric_para_0[:, 2] + self.fric_para_0[:, 4] * (self.dof_vel[:, 0] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 6] <= 0.002) & (self.dof_vel[:, 6] >= -0.002)
        flag_1 = ((self.dof_vel[:, 6] > 0.002) & (self.dof_vel[:, 6] <= 0.16))
        flag_2 = (self.dof_vel[:, 6] > 0.16)
        flag_3 = ((self.dof_vel[:, 6] < -0.002) & (self.dof_vel[:, 6] >= -0.16))
        flag_4 = (self.dof_vel[:, 6] < -0.16)

        self.fric[:, 6] = self.fric_para_0[:, 0] / 0.002 * self.dof_vel[:, 6] * flag_0 + \
                          ((self.fric_para_0[:, 1] - self.fric_para_0[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 6] - 0.002) + self.fric_para_0[:, 0]) * flag_1 + \
                          (self.fric_para_0[:, 1] + self.fric_para_0[:, 3] * (self.dof_vel[:, 6] - 0.16)) * flag_2 + \
                          ((self.fric_para_0[:, 2] + self.fric_para_0[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 6] + 0.002) - self.fric_para_0[:, 0]) * flag_3 + \
                          (self.fric_para_0[:, 2] + self.fric_para_0[:, 4] * (self.dof_vel[:, 6] + 0.16)) * flag_4

    def friction_1_7(self):
        flag_0 = (self.dof_vel[:, 1] <= 0.002) & (self.dof_vel[:, 1] >= -0.002)
        flag_1 = ((self.dof_vel[:, 1] > 0.002) & (self.dof_vel[:, 1] <= 0.16))
        flag_2 = (self.dof_vel[:, 1] > 0.16)
        flag_3 = ((self.dof_vel[:, 1] < -0.002) & (self.dof_vel[:, 1] >= -0.16))
        flag_4 = (self.dof_vel[:, 1] < -0.16)

        self.fric[:, 1] = self.fric_para_1[:, 0] / 0.002 * self.dof_vel[:, 1] * flag_0 + \
                          ((self.fric_para_1[:, 1] - self.fric_para_1[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 1] - 0.002) + self.fric_para_1[:, 0]) * flag_1 + \
                          (self.fric_para_1[:, 1] + self.fric_para_1[:, 3] * (self.dof_vel[:, 1] - 0.16)) * flag_2 + \
                          ((self.fric_para_1[:, 2] + self.fric_para_1[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 1] + 0.002) - self.fric_para_1[:, 0]) * flag_3 + \
                          (self.fric_para_1[:, 2] + self.fric_para_1[:, 4] * (self.dof_vel[:, 1] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 7] <= 0.002) & (self.dof_vel[:, 7] >= -0.002)
        flag_1 = ((self.dof_vel[:, 7] > 0.002) & (self.dof_vel[:, 7] <= 0.16))
        flag_2 = (self.dof_vel[:, 7] > 0.16)
        flag_3 = ((self.dof_vel[:, 7] < -0.002) & (self.dof_vel[:, 7] >= -0.16))
        flag_4 = (self.dof_vel[:, 7] < -0.16)

        self.fric[:, 7] = self.fric_para_1[:, 0] / 0.002 * self.dof_vel[:, 7] * flag_0 + \
                          ((self.fric_para_1[:, 1] - self.fric_para_1[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 7] - 0.002) + self.fric_para_1[:, 0]) * flag_1 + \
                          (self.fric_para_1[:, 1] + self.fric_para_1[:, 3] * (self.dof_vel[:, 7] - 0.16)) * flag_2 + \
                          ((self.fric_para_1[:, 2] + self.fric_para_1[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 7] + 0.002) - self.fric_para_1[:, 0]) * flag_3 + \
                          (self.fric_para_1[:, 2] + self.fric_para_1[:, 4] * (self.dof_vel[:, 7] + 0.16)) * flag_4

    def friction_2_3_8_9(self):
        flag_0 = (self.dof_vel[:, 2] <= 0.002) & (self.dof_vel[:, 2] >= -0.002)
        flag_1 = ((self.dof_vel[:, 2] > 0.002) & (self.dof_vel[:, 2] <= 0.16))
        flag_2 = (self.dof_vel[:, 2] > 0.16)
        flag_3 = ((self.dof_vel[:, 2] < -0.002) & (self.dof_vel[:, 2] >= -0.16))
        flag_4 = (self.dof_vel[:, 2] < -0.16)

        self.fric[:, 2] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 2] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 2] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 2] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 2] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 2] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 3] <= 0.002) & (self.dof_vel[:, 3] >= -0.002)
        flag_1 = ((self.dof_vel[:, 3] > 0.002) & (self.dof_vel[:, 3] <= 0.16))
        flag_2 = (self.dof_vel[:, 3] > 0.16)
        flag_3 = ((self.dof_vel[:, 3] < -0.002) & (self.dof_vel[:, 3] >= -0.16))
        flag_4 = (self.dof_vel[:, 3] < -0.16)

        self.fric[:, 3] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 3] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 3] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 3] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 3] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 3] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 8] <= 0.002) & (self.dof_vel[:, 8] >= -0.002)
        flag_1 = ((self.dof_vel[:, 8] > 0.002) & (self.dof_vel[:, 8] <= 0.16))
        flag_2 = (self.dof_vel[:, 8] > 0.16)
        flag_3 = ((self.dof_vel[:, 8] < -0.002) & (self.dof_vel[:, 8] >= -0.16))
        flag_4 = (self.dof_vel[:, 8] < -0.16)

        self.fric[:, 8] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 8] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 8] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 8] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 8] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 8] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 9] <= 0.002) & (self.dof_vel[:, 9] >= -0.002)
        flag_1 = ((self.dof_vel[:, 9] > 0.002) & (self.dof_vel[:, 9] <= 0.16))
        flag_2 = (self.dof_vel[:, 9] > 0.16)
        flag_3 = ((self.dof_vel[:, 9] < -0.002) & (self.dof_vel[:, 9] >= -0.16))
        flag_4 = (self.dof_vel[:, 9] < -0.16)

        self.fric[:, 9] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 9] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 9] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 9] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 9] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 9] + 0.16)) * flag_4

    # ======================================================================================================================
    # Reward functions
    # ======================================================================================================================

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_body_states[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        pos_target[stand_command] = self.default_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[stand_command] = 1.0
        return r

    def _reward_tracking_lin_vel_exp(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error_square = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_abs = torch.sum(torch.abs(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_square = torch.exp(-lin_vel_error_square / self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-lin_vel_error_abs * 2 / self.cfg.rewards.tracking_sigma)
        r = torch.where(stand_command, r_abs, r_square)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # r = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        return r

    def _reward_tracking_ang_vel(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        ang_vel_error_square = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_abs = torch.abs(
            self.commands[:, 2] - self.base_lin_vel[:, 2])
        r_square = torch.exp(-ang_vel_error_square / self.cfg.rewards.tracking_sigma_ang)
        r_abs = torch.exp(-ang_vel_error_abs * 2 / self.cfg.rewards.tracking_sigma_ang)
        r = torch.where(stand_command, r_abs, r_square)
        # ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # r = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma_ang)
        return r

    def _reward_stand_still(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r

    # def _reward_roll_pitch(self):
    #     stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     roll_pitch = torch.norm(torch.cat([self.pitch, self.roll], dim=-1))
    #     r = torch.sum(torch.square(roll_pitch[stand_command]))
    #     return r
    # def _reward_stand_vel(self):
    #     stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     roll_pitch = torch.norm(torch.cat([self.pitch, self.roll], dim=-1))
    #     r = torch.sum(torch.square(roll_pitch[stand_command]))
    #     return r
    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)