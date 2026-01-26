import torch
import torch_directml
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = CIFAR10CNN()
    state_dict = torch.load("./train_models/cifar10_cnn_20.pth")
    model.load_state_dict(state_dict)

    cat_img = Image.open("./ship.jpg")
    trans = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    cat_tensor = trans(cat_img)
    cat_tensor = cat_tensor.unsqueeze(0)
    print(cat_tensor.shape)
    test_dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transforms.ToTensor()
    )

    labels = test_dataset.classes

    # 前向传播
    with torch.no_grad():
        model.eval()
        outputs = model(cat_tensor)
        predicted = outputs.argmax(dim=1)
        print(labels[predicted.item()])
        # ship
