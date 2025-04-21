import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import ast

# 设置中文字体，请确保你的系统中安装了相应的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def read_csv_to_array(filename):
    df = pd.read_csv(filename, header=None)

    data = np.array([ast.literal_eval(row[0]) for _, row in df.iterrows()])
    return data

def plot_ang_vel(ang_vel_file):
    ang_vel = read_csv_to_array(ang_vel_file)

    ang_x = ang_vel[:, 0]
    ang_y = ang_vel[:, 1]
    ang_z = ang_vel[:, 2]

    # 创建时间步
    time_steps = np.arange(len(ang_x))

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 左侧关节角度
    plt.plot(time_steps, ang_x, label='横滚角角速度')
    plt.plot(time_steps, ang_y, label='俯仰角角速度')
    plt.plot(time_steps, ang_z, label='偏航角角速度')

    # ax1.set_title('Left Joint Angles over Time Steps')
    plt.xlabel('时间步长',fontsize=20)
    plt.ylabel('弧度 (rad)',fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.legend(fontsize=18, loc='best')
    plt.tick_params(axis='both', labelsize=20)

    plt.show()

    # 保存图表
    # plt.savefig('ang_vel.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    plot_ang_vel('Ang_vel.csv')