"""
第5章 误差反向传播法 - 完整代码
包含：计算图、链式法则、完整反向传播实现
"""

import numpy as np


# ========== 5.1 计算图 ==========


class MulLayer:
    """乘法层"""

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    """加法层"""

    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# ========== 5.2 链式法则 ==========


class ReluLayer:
    """ReLU 层"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class SigmoidLayer:
    """Sigmoid 层"""

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


# ========== 5.3 完整的反向传播实现 ==========


def softmax(a):
    """Softmax 函数（数值稳定版）"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


def cross_entropy_error(y, t):
    """交叉熵误差"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


class AffineLayer:
    """矩阵乘法层（Affine 层）"""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLossLayer:
    """Softmax + 交叉熵损失层"""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# ========== 测试代码 ==========

if __name__ == "__main__":
    # === 5.1 计算图测试 ===
    print("=== 5.1 计算图 ===")
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_fruit_layer = AddLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    total_price = add_fruit_layer.forward(apple_price, orange_price)
    final_price = mul_tax_layer.forward(total_price, tax)

    print("最终价格:", final_price)  # (100*2 + 150*3) * 1.1 = 715

    dprice = 1.0
    dtotal_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_fruit_layer.backward(dtotal_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)

    print("dapple =", dapple)
    print("dapple_num =", dapple_num)
    print("dorange =", dorange)
    print("dorange_num =", dorange_num)
    print("dtax =", dtax)

    # === 5.2 链式法则测试 ===
    print("\n=== 5.2 链式法则 ===")
    relu = ReluLayer()
    x = np.array([[1, -2], [-3, 4]])
    y = relu.forward(x)
    print("ReLU 正向输出:", y)
    dy = relu.backward(np.ones_like(y))
    print("ReLU 反向输出:", dy)

    sigmoid = SigmoidLayer()
    y = sigmoid.forward(x)
    print("Sigmoid 正向输出:", y)
    dy = sigmoid.backward(np.ones_like(y))
    print("Sigmoid 反向输出:", dy)

    # === 5.3 完整反向传播测试 ===
    print("\n=== 5.3 完整反向传播 ===")
    W = np.array([[0.1, 0.2], [0.3, 0.4]])
    b = np.array([0.1, 0.2])
    affine = AffineLayer(W, b)

    x = np.array([[1.0, 2.0]])
    out = affine.forward(x)
    print("Affine 正向输出:", out)

    dout = np.array([[1.0, 1.0]])
    dx = affine.backward(dout)
    print("Affine 反向 dx:", dx)
    print("Affine 反向 dW:", affine.dW)
    print("Affine 反向 db:", affine.db)

    # SoftmaxWithLoss 测试
    print("\n=== SoftmaxWithLoss 测试 ===")
    softmax_loss = SoftmaxWithLossLayer()
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    t = np.array([[0, 1, 0], [1, 0, 0]])

    loss = softmax_loss.forward(x, t)
    print("损失值:", loss)

    dx = softmax_loss.backward()
    print("反向传播输出:", dx)
    print("形状:", dx.shape)
