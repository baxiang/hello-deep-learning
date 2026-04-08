import numpy as np


class SGD:
    """随机梯度下降"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """动量优化器"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """自适应学习率优化器"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key in params.keys():
                self.h[key] = np.zeros_like(params[key])

        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """Adam 优化器"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


if __name__ == "__main__":
    # 测试优化器
    class Func:
        def f(self, x):
            return x[0] ** 2 / 20.0 + x[1] ** 2

    func = Func()

    for opt_name in ["SGD", "Momentum", "AdaGrad", "Adam"]:
        if opt_name == "SGD":
            optimizer = SGD(lr=0.95)
        elif opt_name == "Momentum":
            optimizer = Momentum(lr=0.1)
        elif opt_name == "AdaGrad":
            optimizer = AdaGrad(lr=1.5)
        elif opt_name == "Adam":
            optimizer = Adam(lr=0.3)

        x = np.array([-7.0, 2.0])
        print(f"\n{opt_name}:")
        print(f"  初始位置: {x}")

        for i in range(30):
            grad = np.array([x[0] / 10.0, 2 * x[1]])
            optimizer.update({"x": x}, {"x": grad})

        print(f"  最终位置: {x}")
        print(f"  函数值: {func.f(x):.6f}")
