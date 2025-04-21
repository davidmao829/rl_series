import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 1. 数据加载和预处理
def load_and_preprocess_data(filename, window_size=3):
    """加载数据并创建时间窗口特征"""
    data = pd.read_csv(filename, header=None)
    raw_data = data.values

    # 提取各列数据
    hip_torque = raw_data[:, 0]  # 髋关节实际力矩
    knee_torque = raw_data[:, 1]  # 膝关节实际力矩
    hip_error = raw_data[:, 2]  # 髋关节误差
    knee_error = raw_data[:, 3]  # 膝关节误差
    hip_velocity = raw_data[:, 4]  # 髋关节速度
    knee_velocity = raw_data[:, 5]  # 膝关节速度

    # 创建时间窗口特征
    def create_sequences(data, targets, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(targets[i + window_size])
        return np.array(X), np.array(y)

    # 髋关节数据
    hip_features = np.column_stack((hip_error, hip_velocity))
    X_hip, y_hip = create_sequences(hip_features, hip_torque, window_size)

    # 膝关节数据
    knee_features = np.column_stack((knee_error, knee_velocity))
    X_knee, y_knee = create_sequences(knee_features, knee_torque, window_size)

    return X_hip, y_hip, X_knee, y_knee


# 2. 定义神经网络模型
class TorqueModel(nn.Module):
    def __init__(self, input_size):
        super(TorqueModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加Dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加Dropout
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# 3. 训练函数（添加模型保存功能）
def train_model(model, X_train, y_train, X_test, y_test, model_name, epochs=100, batch_size=32):
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    train_losses, test_losses = [], []
    best_test_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # 计算训练和测试损失
        train_loss = np.mean(batch_losses)
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            print(f'Epoch {epoch + 1}: 保存最佳模型，测试损失: {test_loss:.4f}')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return model, train_losses, test_losses


# 4. 评估和可视化
def evaluate_and_plot(model, X_test, y_test, joint_name):
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_test)).numpy().flatten()

    # 计算R平方
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:200], 'b-', label='真实值', alpha=0.7)
    plt.plot(predictions[:200], 'r--', label='预测值', alpha=0.7)
    plt.title(f'{joint_name}力矩预测对比 (R²={r2:.3f})')
    plt.xlabel('时间步')
    plt.ylabel('力矩 (Nm)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return r2


# 5. 新增验证函数
def validate_saved_models():
    """加载已保存的模型并进行验证"""
    # 确保测试数据存在
    if not os.path.exists('test_data'):
        print("错误：找不到测试数据目录 'test_data'")
        return

    try:
        # 加载测试数据
        X_hip_test = np.load('test_data/X_hip_test.npy')
        y_hip_test = np.load('test_data/y_hip_test.npy')
        X_knee_test = np.load('test_data/X_knee_test.npy')
        y_knee_test = np.load('test_data/y_knee_test.npy')
    except:
        print("错误：加载测试数据失败，请先运行训练过程")
        return

    # 加载髋关节模型
    if os.path.exists('hip_best.pth'):
        hip_model = TorqueModel(input_size=6)
        hip_model.load_state_dict(torch.load('hip_best.pth'))
        hip_model.eval()
        print("\n验证髋关节模型...")
        evaluate_and_plot(hip_model, X_hip_test, y_hip_test, "髋关节")
    else:
        print("警告：找不到髋关节模型文件 'hip_best.pth'")

    # 加载膝关节模型
    if os.path.exists('knee_best.pth'):
        knee_model = TorqueModel(input_size=6)
        knee_model.load_state_dict(torch.load('knee_best.pth'))
        knee_model.eval()
        print("\n验证膝关节模型...")
        evaluate_and_plot(knee_model, X_knee_test, y_knee_test, "膝关节")
    else:
        print("警告：找不到膝关节模型文件 'knee_best.pth'")


# 主函数
def main():
    # 加载数据
    X_hip, y_hip, X_knee, y_knee = load_and_preprocess_data('Torque_Data.csv')

    # 划分训练集和测试集
    X_hip_train, X_hip_test, y_hip_train, y_hip_test = train_test_split(
        X_hip, y_hip, test_size=0.2, random_state=42)
    X_knee_train, X_knee_test, y_knee_train, y_knee_test = train_test_split(
        X_knee, y_knee, test_size=0.2, random_state=42)

    # 创建测试数据目录
    os.makedirs('test_data', exist_ok=True)

    # 保存测试数据
    np.save('test_data/X_hip_test.npy', X_hip_test.reshape(-1, 6))
    np.save('test_data/y_hip_test.npy', y_hip_test)
    np.save('test_data/X_knee_test.npy', X_knee_test.reshape(-1, 6))
    np.save('test_data/y_knee_test.npy', y_knee_test)

    # 训练髋关节模型
    print("\n训练髋关节模型...")
    hip_model = TorqueModel(input_size=6)
    hip_model, hip_train_loss, hip_test_loss = train_model(
        hip_model, X_hip_train.reshape(-1, 6), y_hip_train,
        X_hip_test.reshape(-1, 6), y_hip_test, 'hip', epochs=500)

    # 训练膝关节模型
    print("\n训练膝关节模型...")
    knee_model = TorqueModel(input_size=6)
    knee_model, knee_train_loss, knee_test_loss = train_model(
        knee_model, X_knee_train.reshape(-1, 6), y_knee_train,
        X_knee_test.reshape(-1, 6), y_knee_test, 'knee', epochs=500)

    # 评估模型
    print("\n评估髋关节模型...")
    hip_r2 = evaluate_and_plot(hip_model, X_hip_test.reshape(-1, 6), y_hip_test, "髋关节")

    print("\n评估膝关节模型...")
    knee_r2 = evaluate_and_plot(knee_model, X_knee_test.reshape(-1, 6), y_knee_test, "膝关节")

    # 绘制训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(hip_train_loss, label='髋关节训练损失')
    plt.plot(hip_test_loss, label='髋关节测试损失')
    plt.plot(knee_train_loss, label='膝关节训练损失')
    plt.plot(knee_test_loss, label='膝关节测试损失')
    plt.title('训练和测试损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    os.makedirs('test_data', exist_ok=True)

    # 检查是否要运行训练还是仅验证
    if len(os.listdir('test_data')) > 0 and os.path.exists('hip_best.pth') and os.path.exists('knee_best.pth'):
        answer = input("检测到已有训练好的模型和测试数据，是否只进行验证？(y/n): ")
        if answer.lower() == 'y':
            validate_saved_models()
        else:
            main()
    else:
        main()