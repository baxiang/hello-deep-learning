import numpy as np

# 创建一维数组
x = np.array([1.0, 2.0, 3.0])
print("x =", x)
print("x 的形状:", x.shape)

# 创建二维数组
A = np.array([[1, 2], [3, 4]])
print("\nA =")
print(A)
print("A 的形状:", A.shape)

# 数组四则运算
B = np.array([[3, 0], [0, 6]])
print("\nA + B =")
print(A + B)

print("\nA * B（逐元素相乘）=")
print(A * B)

# 广播
print("\nA * 10（标量广播）=")
print(A * 10)

C = np.array([10, 20])
print("\nA * C（行向量广播）=")
print(A * C)

# 访问元素
print("\nA[0] =", A[0])
print("A[0][1] =", A[0][1])
