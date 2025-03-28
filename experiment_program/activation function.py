import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

def plot_activation_function():
    x = np.linspace(-4,4,100)

    relu = np.maximum(0, x)
    tanh = np.tanh(x)

    plt.figure(figsize=(10,8))
    plt.plot(x, relu, label='ReLU', color='blue', linewidth=2)
    plt.plot(x, tanh, label='Tanh', color='orange', linewidth=2)

    # plt.title('激活函数')
    plt.xlabel('x', fontsize=20)
    plt.ylabel('f(x)', fontsize=20)

    plt.ylim(-2,2)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()

    plt.tick_params(axis='both', labelsize=16)

    plt.legend(loc='upper right', fontsize=20)
    # plt.show()
    plt.savefig('activation_func')

if __name__ == '__main__':
    plot_activation_function()