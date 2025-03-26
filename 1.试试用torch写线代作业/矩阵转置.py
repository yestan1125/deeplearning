import torch

# 创建一个2x3的浮点型矩阵
matrix = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float32)

# 方法1：使用t()函数进行转置
transpose_matrix = matrix.t()

# 方法2：使用transpose函数进行转置，指定维度0和1交换
transpose_matrix2 = torch.transpose(matrix, 0, 1)

# 方法3：使用permute函数进行转置，重新排列维度
transpose_matrix3 = matrix.permute(1, 0)

# 方法4：使用view函数进行转置，将矩阵视为一维向量，然后重新排列维度
transpose_matrix4 = matrix.view(-1).view(matrix.size()[1], matrix.size()[0])

# 方法5：使用numpy的transpose函数进行转置，然后将结果转换为PyTorch张量
transpose_matrix5 = torch.from_numpy(matrix.numpy().transpose())

# 方法6：使用reshape函数进行转置，将矩阵视为一维向量，然后重新排列维度
transpose_matrix6 = matrix.reshape(-1).reshape(matrix.size()[1], matrix.size()[0])

# 方法7：使用split函数进行转置，将矩阵拆分为行向量，然后重新排列维度
transpose_matrix7 = torch.stack(matrix.split(1, dim=0), dim=1).squeeze()

# 方法8：使用einsum函数进行转置，使用Einstein求和约定进行矩阵转置
transpose_matrix8 = torch.einsum('ij->ji', matrix)

# 方法9：使用unsqueeze和transpose函数进行转置，将矩阵视为一维向量，然后重新排列维度
transpose_matrix9 = matrix.unsqueeze(0).transpose(0, 2).squeeze()

# 方法10：使用expand和permute函数进行转置，将矩阵视为一维向量，然后重新排列维度
transpose_matrix10 = matrix.expand(matrix.size()[1], matrix.size()[0]).permute(1, 0)

# 打印结果
print("原始矩阵：\n", matrix)
print("转置矩阵（方法1）：\n", transpose_matrix)
print("转置矩阵（方法2）：\n", transpose_matrix2)
print("转置矩阵（方法3）：\n", transpose_matrix3)
print("转置矩阵（方法4）：\n", transpose_matrix4)
print("转置矩阵（方法5）：\n", transpose_matrix5)
print("转置矩阵（方法6）：\n", transpose_matrix6)
print("转置矩阵（方法7）：\n", transpose_matrix7)
print("转置矩阵（方法8）：\n", transpose_matrix8)
print("转置矩阵（方法9）：\n", transpose_matrix9)
print("转置矩阵（方法10）：\n", transpose_matrix10)
