import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def read_joint_data(filename, num_steps=100):
    """读取关节数据文件，返回前num_steps行的数据"""
    df = pd.read_csv(filename, header=None)
    # 读取前num_steps行，每行按逗号分割转换为浮点数
    data = []
    for i in range(min(num_steps, len(df))):
        row = df.iloc[i]
        if isinstance(row[0], str):
            values = [float(x) for x in row[0].split(',')]
        else:
            values = row.values.astype(float)
        data.append(values)
    return np.array(data)


def plot_joint_comparison(ideal_file, actual_file, num_steps=100):
    """绘制理想和实际关节位置对比图"""
    # 读取数据
    ideal_data = read_joint_data(ideal_file, num_steps)
    actual_data = read_joint_data(actual_file, num_steps)

    # 提取髋关节(索引2)和膝关节(索引3)的数据
    # 前14个是实际位置，后14个是目标位置
    hip_idx, knee_idx = 2, 3

    # 理想模型数据
    ideal_hip_actual = ideal_data[:, hip_idx]  # 实际髋关节位置
    ideal_knee_actual = ideal_data[:, knee_idx]  # 实际膝关节位置
    ideal_hip_target = ideal_data[:, hip_idx + 14]  # 目标髋关节位置
    ideal_knee_target = ideal_data[:, knee_idx + 14]  # 目标膝关节位置

    # 实际模型数据
    actual_hip_actual = actual_data[:, hip_idx]  # 实际髋关节位置
    actual_knee_actual = actual_data[:, knee_idx]  # 实际膝关节位置
    actual_hip_target = actual_data[:, hip_idx + 14] / 180 * math.pi  # 目标髋关节位置
    actual_knee_target = actual_data[:, knee_idx + 14] / 180 * math.pi  # 目标膝关节位置

    # 创建时间步
    time_steps = np.arange(num_steps)

    # 创建图表，1行2列
    plt.figure(figsize=(20, 8))

    # 左侧子图 - 理想模型
    plt.subplot(1, 2, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, ideal_hip_actual, 'b-', label='髋关节实际位置')
    plt.plot(time_steps, ideal_hip_target, 'b--', label='髋关节目标位置')
    plt.plot(time_steps, ideal_knee_actual, 'r-', label='膝关节实际位置')
    plt.plot(time_steps, ideal_knee_target, 'r--', label='膝关节目标位置')
    plt.xlabel('时间步长', fontsize=16)
    plt.ylabel('关节角度 (rad)', fontsize=16)
    plt.title('理想电机模型下的关节跟踪', fontsize=18)
    plt.legend(fontsize=12, loc='best')
    plt.tick_params(axis='both', labelsize=14)

    # 右侧子图 - 实际模型
    plt.subplot(1, 2, 2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, actual_hip_actual, 'b-', label='髋关节实际位置')
    plt.plot(time_steps, actual_hip_target, 'b--', label='髋关节目标位置')
    plt.plot(time_steps, actual_knee_actual, 'r-', label='膝关节实际位置')
    plt.plot(time_steps, actual_knee_target, 'r--', label='膝关节目标位置')
    plt.xlabel('时间步长', fontsize=16)
    plt.ylabel('关节角度 (rad)', fontsize=16)
    plt.title('实际电机模型下的关节跟踪', fontsize=18)
    plt.legend(fontsize=12, loc='best')
    plt.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.show()


def plot_torque_comparison(ideal_file, num_steps=100):
    """绘制理想和实际关节位置对比图"""
    # 读取数据
    torque_data = read_joint_data(ideal_file, num_steps)

    # 提取髋关节(索引2)和膝关节(索引3)的数据
    # 前14个是实际位置，后14个是目标位置
    # left_hip_idx, left_knee_idx, right_hip_idx, right_knee_idx = 2, 3, 8, 9
    left_hip_idx, left_knee_idx = 0, 1

    # 实际模型数据
    actual_left_hip = torque_data[:, left_hip_idx]  # 实际左髋关节力矩
    actual_left_knee = torque_data[:, left_knee_idx]  # 实际左膝关节力矩
    # actual_right_hip = torque_data[:, right_hip_idx]  # 实际右髋关节力矩
    # actual_right_knee = torque_data[:, right_knee_idx]  # 实际右膝关节力矩

    # 理想模型数据
    # ideal_left_hip = torque_data[:, left_hip_idx + 14]  / 2.0# 理想左髋关节力矩
    # ideal_left_knee = torque_data[:, left_knee_idx + 14] / 2.0 # 理想左膝关节力矩
    ideal_left_hip = torque_data[:, left_hip_idx + 2] / 1.5  # 理想左髋关节力矩
    ideal_left_knee = torque_data[:, left_knee_idx + 2] / 2  # 理想左膝关节力矩
    # ideal_right_hip = torque_data[:, right_hip_idx + 14]  # 理想右髋关节力矩
    # ideal_right_knee = torque_data[:, right_knee_idx + 14]  # 理想右膝关节力矩

    # 创建时间步
    time_steps = np.arange(num_steps)

    # 创建图表，1行2列
    plt.figure(figsize=(20, 8))

    # 左侧子图 - 理想模型
    plt.subplot(1, 2, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, actual_left_hip, 'b-', label='髋关节实际力矩')
    plt.plot(time_steps, ideal_left_hip, 'r--', label='膝关节目标力矩')
    plt.xlabel('时间步长', fontsize=16)
    plt.ylabel('关节力矩 (N·m)', fontsize=16)
    plt.title('左腿髋关节力矩跟踪', fontsize=18)
    plt.legend(fontsize=12, loc='best')
    plt.tick_params(axis='both', labelsize=14)

    # 右侧子图 - 实际模型
    plt.subplot(1, 2, 2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, actual_left_knee, 'b-', label='膝关节实际力矩')
    plt.plot(time_steps, ideal_left_knee, 'r--', label='膝关节目标力矩')
    plt.xlabel('时间步长', fontsize=16)
    plt.ylabel('关节力矩 (N·m)', fontsize=16)
    plt.title('右腿髋关节力矩跟踪', fontsize=18)
    plt.legend(fontsize=12, loc='best')
    plt.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 可以在这里修改要显示的时间步数
    # plot_joint_comparison('Ideal_Swing_test.csv', 'Actual_Swing_test.csv', num_steps=350)
    # plot_torque_comparison('Ideal_torque.csv', num_steps=200)
    # plot_torque_comparison('Actual_torque.csv', num_steps=500)
    # plot_torque_comparison('Actual_torque_1.6.csv', num_steps=500)
    plot_torque_comparison('Torque_Data.csv', num_steps=500)

