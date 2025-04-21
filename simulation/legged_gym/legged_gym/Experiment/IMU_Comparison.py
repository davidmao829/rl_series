import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import ast
import math
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def read_csv_to_array(filename):
    df = pd.read_csv(filename, header=None)
    data = np.array([ast.literal_eval(row[0]) for _, row in df.iterrows()])
    return data


def read_obs_csv(filename):
    df = pd.read_csv(filename, header=None)
    # 处理两种可能的格式：字符串或数值
    obs_data = []
    for _, row in df.iterrows():
        if isinstance(row[0], str):
            # 如果是字符串格式，按逗号分割
            values = [float(x) for x in row[0].split(',')]
        else:
            # 如果已经是数值格式，直接取所有列
            values = row.values.astype(float)
        obs_data.append(values[5:8])  # 取索引5,6,7（第6,7,8个数据）
    return np.array(obs_data)



def plot_comparison(ang_vel_file, obs_file, num_steps=100):
    ang_vel = read_csv_to_array(ang_vel_file)
    obs_data = read_obs_csv(obs_file)

    # 限制数据点数量
    ang_vel = ang_vel[:num_steps]
    obs_data = obs_data[:num_steps]

    ang_x = ang_vel[:, 0]
    ang_y = ang_vel[:, 1]
    ang_z = ang_vel[:, 2]

    obs_5 = obs_data[:, 0] /math.pi * 180
    obs_6 = obs_data[:, 1] /math.pi * 180
    obs_7 = obs_data[:, 2] /math.pi * 180

    # 创建时间步
    time_steps = np.arange(num_steps)

    # 创建图表，现在有两个子图
    plt.figure(figsize=(16, 6))

    # 左侧子图 - 原始角速度数据
    plt.subplot(1, 2, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, ang_x, label='仿真横滚角角速度')
    plt.plot(time_steps, ang_y, label='仿真俯仰角角速度')
    plt.plot(time_steps, ang_z, label='仿真偏航角角速度')
    plt.xlabel('时间步长', fontsize=20)
    plt.ylabel('角速度 (degree/s)', fontsize=20)
    plt.legend(fontsize=16, loc='upper left')
    plt.tick_params(axis='both', labelsize=20)
    plt.title(f'仿真角速度数据', fontsize=20)

    # 右侧子图 - 观测数据
    plt.subplot(1, 2, 2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, obs_5, label='实际横滚角角速度')
    plt.plot(time_steps, obs_6, label='实际俯仰角角速度')
    plt.plot(time_steps, obs_7, label='实际偏航角角速度')
    plt.xlabel('时间步长', fontsize=20)
    plt.ylabel('角速度(degree/s)', fontsize=20)
    plt.legend(fontsize=16, loc='best')
    plt.tick_params(axis='both', labelsize=20)
    plt.title(f'实际角速度数据', fontsize=20)

    plt.tight_layout()
    # plt.show()
    plt.savefig('IMU_Comparison.jpg', dpi=500)

if __name__ == '__main__':
    # 可以在这里修改要显示的时间步数
    plot_comparison('Ang_vel.csv', 'obs_for_ang_vel.csv', num_steps=100)