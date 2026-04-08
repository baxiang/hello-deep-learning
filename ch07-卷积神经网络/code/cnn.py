import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


class SimpleConvNet:
    """简单的卷积神经网络"""

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "stride": 1, "pad": 0},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):

        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_stride = conv_param["stride"]
        filter_pad = conv_param["pad"]

        input_c = input_dim[0]
        input_h = input_dim[1]
        input_w = input_dim[2]

        conv_out_h = (input_h + 2 * filter_pad - filter_size) // filter_stride + 1
        conv_out_w = (input_w + 2 * filter_pad - filter_size) // filter_stride + 1
        pool_out_h = conv_out_h // 2
        pool_out_w = conv_out_w // 2
        fc_input_size = pool_out_h * pool_out_w * filter_num

        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_c, filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(
            fc_input_size, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        self.filter_num = filter_num
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.filter_pad = filter_pad
        self.pool_size = 2
        self.pool_stride = 2
        self.fc_input_size = fc_input_size

    def conv_simple(self, x, w, b, stride, pad):
        n, c, h, w_x = x.shape
        f_n, f_c, f_h, f_w = w.shape
        out_h = (h + 2 * pad - f_h) // stride + 1
        out_w = (w_x + 2 * pad - f_w) // stride + 1

        out = np.zeros((n, f_n, out_h, out_w))
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                x_slice = x_padded[
                    :, :, h_start : h_start + f_h, w_start : w_start + f_w
                ]
                for f in range(f_n):
                    out[:, f, i, j] = np.sum(x_slice * w[f], axis=(1, 2, 3)) + b[f]
        return out

    def pool_simple(self, x, pool_size, pool_stride):
        n, c, h, w = x.shape
        out_h = (h - pool_size) // pool_stride + 1
        out_w = (w - pool_size) // pool_stride + 1

        out = np.zeros((n, c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * pool_stride
                w_start = j * pool_stride
                x_slice = x[
                    :, :, h_start : h_start + pool_size, w_start : w_start + pool_size
                ]
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))
        return out

    def predict(self, x):
        W1, W2, W3 = self.params["W1"], self.params["W2"], self.params["W3"]
        b1, b2, b3 = self.params["b1"], self.params["b2"], self.params["b3"]

        conv_out = self.conv_simple(x, W1, b1, self.filter_stride, self.filter_pad)
        relu_out = relu(conv_out)
        pool_out = self.pool_simple(relu_out, self.pool_size, self.pool_stride)

        n = pool_out.shape[0]
        flat = pool_out.reshape(n, -1)

        affine1 = np.dot(flat, W2) + b2
        relu2 = relu(affine1)
        affine2 = np.dot(relu2, W3) + b3

        y = softmax(affine2)
        return y


if __name__ == "__main__":
    print("创建简单 CNN 网络...")
    net = SimpleConvNet(input_dim=(1, 28, 28))

    print(f"全连接层输入大小: {net.fc_input_size}")
    print(
        f"网络参数数量: W1={net.params['W1'].size}, W2={net.params['W2'].size}, W3={net.params['W3'].size}"
    )

    x_dummy = np.random.randn(5, 1, 28, 28)
    print(f"\n输入形状: {x_dummy.shape}")

    y = net.predict(x_dummy)
    print(f"输出形状: {y.shape}")
    print("\n预测结果:")
    for i in range(5):
        print(f"  图片 {i}: 预测数字 = {np.argmax(y[i])}, 概率 = {np.max(y[i]):.4f}")
