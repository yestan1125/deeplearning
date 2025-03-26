import torch


def compute_adjugate(matrix):
    matrix_size = matrix.size(0)
    adjugate = torch.zeros_like(matrix)
    for i in range(matrix_size):
        for j in range(matrix_size):
            # 生成代数余子式矩阵
            sub_matrix = torch.cat([torch.cat([matrix[:i, :j], matrix[:i, j + 1:]], dim=1),
                                    torch.cat([matrix[i + 1:, :j], matrix[i + 1:, j + 1:]], dim=1)], dim=0)
            # 计算代数余子式
            cofactor = (-1) ** (i + j) * torch.det(sub_matrix)
            # 填充伴随矩阵（注意转置）
            adjugate[j, i] = cofactor
    return adjugate


# 示例矩阵
example_matrix = torch.tensor([[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 10.]], dtype=torch.float32)

adjugate_matrix = compute_adjugate(example_matrix)
print("原矩阵:")
print(example_matrix)
print("伴随矩阵:")
print(adjugate_matrix)
    