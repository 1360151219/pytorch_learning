import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# 下载预训练的 ResNet - 50 模型
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

print(model)