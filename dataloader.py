from torchvision import transforms,datasets
from torch.utils.data import DataLoader

def main():
    # 加载训练集
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True)
    print(train_dataset[0])
    print(train_dataset.classes)
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # (<PIL.Image.Image image mode=RGB size=32x32 at 0x110A23CB0>, 6)

if __name__ == "__main__":
    main()
