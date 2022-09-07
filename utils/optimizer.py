from torch.optim import SGD
import torch.nn as nn
import torch

embed = nn.Embedding(5, 8)
print(list(embed.parameters())[0])
list(embed.parameters())[0].grad = torch.ones(5, 8) * 0.01
# print(list(embed.parameters())[0].grad)


optimizer = SGD(embed.parameters(), lr=1, momentum=0)
optimizer.step()
print(list(embed.parameters())[0])