import torch

# 创建一个矩阵
matrix = torch.tensor([[1., 2.], [3., 4.]])

try:
    # 求矩阵的逆
    inverse_matrix = torch.inverse(matrix)
    print("原矩阵:")
    print(matrix)
    print("逆矩阵:")
    print(inverse_matrix)
except RuntimeError as e:
    print(f"错误: {e}")
    