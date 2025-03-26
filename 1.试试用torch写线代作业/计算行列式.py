import torch 

matrix = torch.tensor([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0], 
                      [7.0, 8.0, 10.0]])

determinant = torch.linalg.det(matrix)

print("matrix:", matrix)
print("determinant:", determinant)