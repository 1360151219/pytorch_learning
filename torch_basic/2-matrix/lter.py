import torch


features = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])

labels = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

indices = [3, 0, 4, 1, 2]  # 假设打乱后的索引
batch_size = 2

for i in range(0, len(features), batch_size):
    batch_indices = indices[i : min(i + batch_size, len(features))]
    print(features[batch_indices], labels[batch_indices])
