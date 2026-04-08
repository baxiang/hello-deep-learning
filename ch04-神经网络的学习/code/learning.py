import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    if a.ndim == 1:
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a)
    else:
        c = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a, axis=1, keepdims=True)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    x_flat = x.flatten()
    for i in range(x_flat.size):
        tmp_val = x_flat[i]
        x_flat[i] = tmp_val + h
        x[:] = x_flat.reshape(x.shape)
        fxh1 = f(x)
        x_flat[i] = tmp_val - h
        x[:] = x_flat.reshape(x.shape)
        fxh2 = f(x)
        grad_flat = grad.flatten()
        grad_flat[i] = (fxh1 - fxh2) / (2 * h)
        grad[:] = grad_flat.reshape(grad.shape)
        x_flat[i] = tmp_val
    return grad


class TwoLayerNet:
    """两层神经网络"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
        return grads

    def update_params(self, grads, learning_rate=0.1):
        for key in ("W1", "b1", "W2", "b2"):
            self.params[key] -= learning_rate * grads[key]


if __name__ == "__main__":
    np.random.seed(42)
    # 使用小网络演示（数值微分在大网络上非常慢）
    net = TwoLayerNet(input_size=10, hidden_size=8, output_size=3)

    x_dummy = np.random.randn(20, 10)
    t_dummy = np.zeros((20, 3))
    for i in range(20):
        t_dummy[i, i % 3] = 1

    loss_before = net.loss(x_dummy, t_dummy)
    print("训练前损失: {:.4f}".format(loss_before))

    grads = net.numerical_gradient(x_dummy, t_dummy)
    net.update_params(grads, learning_rate=0.1)

    loss_after = net.loss(x_dummy, t_dummy)
    print("训练后损失: {:.4f}".format(loss_after))
    print("损失下降: {:.4f}".format(loss_before - loss_after))
