import os
import queue
from pathlib import Path

from prompt_toolkit.key_binding.bindings.named_commands import self_insert
from pynput import keyboard
import time
import math
import numpy as np
import typer
import csv
from collections import deque
from fourier_grx_client import ControlGroup, RobotClient
import pandas as pd
from ischedule import run_loop, schedule
from scipy.spatial.transform import Rotation as R
from triton.language import dtype

from utils import *
import onnxruntime as ort
import threading
from Joy.joystick import Joystick

joint_target_queue = queue.Queue()


def init_csv(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path}文件已删除")


class Sim2realCfg:
    num_dofs = 14
    num_actions = 12
    n_proprio = 2 + 3 + 3 + 2 + 2 * (num_dofs) + num_actions
    n_priv_latent = 4 + 1 + 2 * (num_dofs) + 3
    history_len = 10
    control_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13]

    sim_duration = 60
    dt = 0.01
    step_freq = 2
    double_support_threshold = 0.1
    cycle_time = 0.8
    decimation = 10
    action_scale = 0.5
    joint_scale = 0.2

    # action_scale = 0.25

    class obs_scale:
        lin_vel = 1.
        ang_vel = 0.25
        dof_pos = 1.
        dof_vel = 0.05
        quat = 0.5


class DemoNohlaRLWalk:
    """
    Reinforcement Learning Walker
    """

    def __init__(
            self,
            step_freq: int = 500,
            act: bool = True,
            cfg=Sim2realCfg()
    ):
        """
        Initialize the RL Walker

        Input:
        - comm_freq: communication frequency, in Hz
        - step_freq: step frequency, in Hz
        """

        # setup RobotClient
        self.client = RobotClient(namespace="gr/1", server_ip="localhost")
        self.act = act
        time.sleep(1.0)
        self.cfg = cfg
        self.client.enable()
        self.set_gains()
        self.start_time = time.time()
        self.last_target_pos = np.zeros(32)
        self.count_lowlevel = 0
        self.last_action = np.zeros(self.cfg.num_actions)
        self.action = np.zeros(self.cfg.num_actions)
        self.commands = np.zeros(3, dtype=np.float32)
        self.joystick = Joystick()
        # 关节的默认角度
        self.joint_default_position = np.array([
            0.0, 0.0, -0.2, 0.4, -0.2, -0.0,  # left leg (6)
            -0.0, 0.0, -0.2, 0.4, -0.2, 0.0,  # right leg (6)
            0.0, -0.0, 0.0,  # waist (3)
            0.0, 0.0, 0.0,  # waist (3)
            0.0, 0.2, 0.0, -0.52, 0.0, 0.0, 0.0,  # left arm (4)
            0.0, -0.2, 0.0, -0.52, 0.0, 0.0, 0.0,  # right arm (4)
        ])
        self.measured_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 25]

        self.proprio_history_buf = deque(maxlen=cfg.history_len)
        for _ in range(self.cfg.history_len):
            self.proprio_history_buf.append(np.zeros(self.cfg.n_proprio))
        self.priv_latent = np.zeros(self.cfg.n_priv_latent, dtype=np.float32)

        self.last_action = np.zeros(self.cfg.num_actions)
        self.action = np.zeros(self.cfg.num_actions)
        # warm up
        time.sleep(5)

    def GetRemoteStates(self):
        data = self.joystick.read()
        if data is None:
            raise ValueError("Joystick not connected!")
        left_joy, right_joy = data["joystick"]
        btns = data["buttons"]
        # A, B, X, Y
        return left_joy, right_joy, btns

    def set_gains(self):
        """
        Set gains for the robot
        """
        pos_kp = [
            0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
            0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
            1.023, 1.023 * 5, 1.023 * 5,
            0.556, 0.556, 0.556,
            0.556, 0.556, 0.556, 0.556, 0, 0, 0,
            0.556, 0.556, 0.556, 0.556, 0, 0, 0,
        ]

        vel_kp = [
            0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
            0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
            0.03, 0.03, 0.03,
            0.03, 0.03, 0.03,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
        ]
        # fmt: off
        vel_ki = [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]
        pd_kp = [
            251.625, 362.52, 250, 250, 10.0, 0,  # left leg
            251.625, 362.52, 250, 250, 10.0, 0,  # right leg
            # 251.625, 200.52, 300, 300, 10.98, 0,  # left leg
            # 251.625, 200.52, 300, 300, 10.98, 0,  # right leg
            251.625, 251.625 * 5, 251.625 * 5,  # waist
            112.06, 112.06, 112.06,  # head
            62.85, 92.85, 112.06, 112.06, 112.06, 10, 0,  # left arm
            62.85, 92.85, 112.06, 112.06, 112.06, 10, 0,  # right arm
        ]
        pd_kd = [
            14.72, 10.08, 11, 11, 2.0, 0.1,  # left leg
            14.72, 10.08, 11, 11, 2.0, 0.1,  # right leg
            14.72, 14.72, 14.72,  # waist
            3.1, 3.1, 3.1,  # head
            5.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # left arm
            5.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # rig
            # ht arm
        ]
        # pd_kp = [
        #     250., 250.0, 350, 350, 10.98, 0,  # left leg
        #     250., 250.0, 350, 350, 10.98, 0,  # right leg
        #     251.625, 251.625 * 5, 251.625 * 5,  # waist
        #     112.06, 112.06, 112.06,  # head
        #     92.85, 92.85, 112.06, 112.06, 112.06, 10, 10,  # left arm
        #     92.85, 92.85, 112.06, 112.06, 112.06, 10, 10,  # right arm
        # ]
        # pd_kd = [
        #     25, 25, 30, 30, 0.6, 0.1,  # left leg
        #     25, 25, 30, 30, 0.6, 0.1,  # right leg
        #     14.72, 14.72, 14.72,  # waist
        #     3.1, 3.1, 3.1,  # head
        #     2.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # left arm
        #     2.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # rig
        #     # ht arm
        # ]
        # fmt: on
        print(self.client.get_gains())
        self.client.set_gains(position_control_kp=pos_kp, velocity_control_kp=vel_kp, velocity_control_ki=vel_ki,
                              pd_control_kp=pd_kp, pd_control_kd=pd_kd)

    def compute_pd_tau(self, target_q, q, dq):
        pd_kp = np.array([
            251.625, 362.52, 250, 250, 10.98, 11,  # left leg
            251.625, 362.52, 250, 250, 10.98, 11,  # right leg
            50, 50,
        ])
        pd_kd = np.array([
            14.72, 10.08, 20, 20, 0.6, 0.6,  # left leg
            14.72, 10.08, 20, 20, 0.6, 0.6,  # right leg
            5, 5,
        ])
        tau_limit = np.array([48, 60, 160, 160, 16, 8, 48, 60, 160, 160, 16, 8, 38, 38])
        tau = (target_q - q) * pd_kp - pd_kd * dq
        tau_clip = np.clip(tau, -tau_limit, tau_limit)
        return tau_clip

    def sensor_data_thread(self):

        while True:
            global hist_obs, joint_target_position_deg, joint_target_position_deg_fake, previous_command_x, previous_command_y
            start_time = time.time()

            def record_to_csv(obs, path):
                with open(path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for row in obs:
                        formatted_row = [f"{x:.3f}" for x in row]
                        writer.writerow(formatted_row)

            """
            Step function for the RL Walker

            Input:
            - act: whether to actuate the robot or not
            """

            # 获取状态值
            imu_quat = self.client.imu_quaternion

            imu_angular_velocity_deg = self.client.imu_angular_velocity
            imu_angular_velocity = imu_angular_velocity_deg / 180.0 * math.pi  # unit : deg/s

            joint_measured_position_urdf = self.client.joint_positions
            joint_measured_velocity_urdf = self.client.joint_velocities
            imu_euler_ang_deg = self.client.imu_angles
            imu_euler_ang = imu_euler_ang_deg / 180.0 * math.pi

            joint_offset_position = joint_measured_position_urdf[self.measured_index] - self.joint_default_position[
                self.measured_index]

            # parse to numpys
            # 32 dims
            joint_target_position_test = np.rad2deg(self.joint_default_position)
            joint_target_position_deg_fake = np.rad2deg(self.joint_default_position)

            right_leg_phase = math.sin(
                2 * math.pi * self.count_lowlevel * self.cfg.dt / self.cfg.cycle_time)
            left_leg_phase = math.cos(
                2 * math.pi * self.count_lowlevel * self.cfg.dt / self.cfg.cycle_time)

            left_ratio = 0
            right_ratio = 0

            if left_leg_phase < 0:
                left_ratio = max(0, (abs(left_leg_phase) - self.cfg.double_support_threshold) /
                                 (1 - self.cfg.double_support_threshold)) * np.sign(left_leg_phase)

            if right_leg_phase < 0:
                right_ratio = max(0, (abs(right_leg_phase) - self.cfg.double_support_threshold) /
                                 (1 - self.cfg.double_support_threshold)) * np.sign(right_leg_phase)

            joint_target_position_test[2] += np.rad2deg(left_ratio * self.cfg.joint_scale)
            joint_target_position_test[3] += -np.rad2deg(left_ratio * 2 * self.cfg.joint_scale)
            joint_target_position_test[8] += np.rad2deg(right_ratio * self.cfg.joint_scale)
            joint_target_position_test[9] += -np.rad2deg(right_ratio * 2 * self.cfg.joint_scale)

            joint_target_position_test_rad = np.deg2rad(joint_target_position_test)
            # 计算理想力矩
            ideal_torque = self.compute_pd_tau(joint_target_position_test_rad[self.measured_index], joint_measured_position_urdf[self.measured_index]
                                               , joint_measured_velocity_urdf[self.measured_index])[:14]
            cur_tau = self.client.joint_efforts[self.measured_index][:14]
            print("ideal_torque", ideal_torque)
            print("cur_tau", cur_tau)
            joint_pos_error  = joint_target_position_test_rad - joint_measured_position_urdf
            torque_data = np.concatenate([cur_tau[2:4],ideal_torque[2:4],joint_pos_error[2:4],(joint_measured_velocity_urdf[self.measured_index][2:4])])
            # print("joint_target_pos", joint_target_position_test[:14])
            data = np.concatenate([joint_measured_position_urdf[:14],
                                   joint_target_position_test[:14]])
            # print("队列是否为空",joint_target_queue.empty())
            if self.count_lowlevel > 50:
                self.client.move_joints(ControlGroup.ALL, joint_target_position_test, 0.0, degrees=True)
                # record_to_csv(obs.reshape(1, 50), '/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/obs.csv')
                record_to_csv(data.reshape(1,28), '/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/Actual_Swing_test.csv')
                record_to_csv(torque_data.reshape(1,8), '/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/Torque_Data.csv')

            end_time = time.time()
            duration_time = end_time - start_time
            self.count_lowlevel += 1
            if duration_time > 0.01:
                continue
            else:
                time.sleep(0.01 - duration_time)


def main(
        step_freq: int = 500, act: bool = True
):
    walker = DemoNohlaRLWalk(
        step_freq=step_freq, act=act
    )

    thread_sensor_data = threading.Thread(target=walker.sensor_data_thread)
    thread_sensor_data.start()
    thread_sensor_data.join()


if __name__ == "__main__":
    init_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/Actual_Swing_test.csv')
    init_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/Torque_Data.csv')
    typer.run(main)
