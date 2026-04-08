import numpy as np


def sigmoid(x):
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-x))


def softmax(a):
    """Softmax 函数（数值稳定版本）"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def init_network():
    """初始化网络（随机权重，仅用于演示）"""
    network = {}
    network["W1"] = np.random.randn(784, 50) * 0.01
    network["b1"] = np.zeros(50)
    network["W2"] = np.random.randn(50, 100) * 0.01
    network["b2"] = np.zeros(100)
    network["W3"] = np.random.randn(100, 10) * 0.01
    network["b3"] = np.zeros(10)
    return network


def forward(network, x):
    """前向传播"""
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


# 演示：用随机输入模拟推理
x_dummy = np.random.randn(1, 784)  # 模拟一张 28x28 的图片
network = init_network()
y = forward(network, x_dummy)

print("输出层结果（前5个）:", y[0][:5])
print("预测数字:", np.argmax(y[0]))
print("输出总和:", np.sum(y[0]))
