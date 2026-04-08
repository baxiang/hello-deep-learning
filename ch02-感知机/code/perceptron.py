import numpy as np


def AND(x1, x2):
    """与门：两个输入都是 1 时输出 1"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0


def NAND(x1, x2):
    """与非门：两个输入都是 1 时输出 0"""
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0


def OR(x1, x2):
    """或门：至少一个输入是 1 时输出 1"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0


def XOR(x1, x2):
    """异或门：用多层感知机实现"""
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == "__main__":
    print("与门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  AND({x1}, {x2}) = {AND(x1, x2)}")

    print("\n与非门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  NAND({x1}, {x2}) = {NAND(x1, x2)}")

    print("\n或门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  OR({x1}, {x2}) = {OR(x1, x2)}")

    print("\n异或门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  XOR({x1}, {x2}) = {XOR(x1, x2)}")
