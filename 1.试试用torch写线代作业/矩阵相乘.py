import torch
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

result = torch.matmul(a, b)

print("a*b = ", result)