import torch


def get_matrix_rank(matrix, eps=1e-12):
    """
    计算矩阵的秩
    :param matrix: 输入的矩阵
    :param eps: 用于判断奇异值是否为零的阈值
    :return: 矩阵的秩
    """
    # 进行奇异值分解
    _, s, _ = torch.linalg.svd(matrix)
    # 统计大于阈值的奇异值的数量
    rank = torch.sum(s > eps)
    return rank.item()


# 示例
if __name__ == "__main__":
    matrix = torch.tensor([[1, 2, 3], 
                          [4, 5, 6], 
                          [7, 8, 10]], dtype=torch.float32)
    rank = get_matrix_rank(matrix)
    print(f"矩阵的秩为: {rank}")
    