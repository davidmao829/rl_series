import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def read_torque_data(filename, num_steps=100):
    """读取关节数据文件，返回前num_steps行的数据"""
    df = pd.read_csv(filename, header=None)
    # 读取前num_steps行，每行按逗号分割转换为浮点数
    data = []
    for i in range(min(num_steps, len(df))):
        row = df.iloc[i]
        if isinstance(row[0],str):
            values = [float(x) for x in row[0].split(',')]
        else:
            values = row.values.astype(float)
        data.append(values)
    return np.array(data)


def plot_torque_comparison(torque_file, num_steps=100):
    """绘制实际力矩和理想力矩对比图"""
    # 读取数据
    torque_data = read_torque_data(torque_file, num_steps)

    # 提取关键关节数据
    hip_idx, knee_idx = 2, 3  # 左腿髋关节和膝关节索引
    r_hip_idx, r_knee_idx = 8, 9  # 右腿髋关节和膝关节索引

    # 实际力矩值
    actual_hip = torque_data[:, hip_idx]
    actual_knee = torque_data[:, knee_idx]
    actual_r_hip = torque_data[:, r_hip_idx]
    actual_r_knee = torque_data[:, r_knee_idx]

    # 理想力矩值 (后14个数据)
    ideal_hip = torque_data[:, hip_idx + 14]
    ideal_knee = torque_data[:, knee_idx + 14]
    ideal_r_hip = torque_data[:, r_hip_idx + 14]
    ideal_r_knee = torque_data[:, r_knee_idx + 14]

    # 创建时间步
    time_steps = np.arange(len(torque_data))

    # 创建图表，2行2列
    plt.figure(figsize=(15, 10))

    # 左腿髋关节力矩对比
    plt.subplot(2, 2, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, actual_hip, 'b-', label='实际力矩')
    plt.plot(time_steps, ideal_hip, 'r--', label='理想力矩')
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('力矩 (Nm)', fontsize=12)
    plt.title('左腿髋关节力矩对比', fontsize=14)
    plt.legend(fontsize=10)

    # 左腿膝关节力矩对比
    plt.subplot(2, 2, 2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, actual_knee, 'b-', label='实际力矩')
    plt.plot(time_steps, ideal_knee, 'r--', label='理想力矩')
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('力矩 (Nm)', fontsize=12)
    plt.title('左腿膝关节力矩对比', fontsize=14)
    plt.legend(fontsize=10)

    # 右腿髋关节力矩对比
    plt.subplot(2, 2, 3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, actual_r_hip, 'b-', label='实际力矩')
    plt.plot(time_steps, ideal_r_hip, 'r--', label='理想力矩')
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('力矩 (Nm)', fontsize=12)
    plt.title('右腿髋关节力矩对比', fontsize=14)
    plt.legend(fontsize=10)

    # 右腿膝关节力矩对比
    plt.subplot(2, 2, 4)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_steps, actual_r_knee, 'b-', label='实际力矩')
    plt.plot(time_steps, ideal_r_knee, 'r--', label='理想力矩')
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('力矩 (Nm)', fontsize=12)
    plt.title('右腿膝关节力矩对比', fontsize=14)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 使用示例
    plot_torque_comparison('Actual_torque.csv', num_steps=100)