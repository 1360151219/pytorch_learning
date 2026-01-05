from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def main():
    # 加载训练集
    # train_dataset = datasets.CIFAR10(root="./dataset", train=True, download=True)
    # print(train_dataset.classes)
    # print(train_dataset[0])
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # (<PIL.Image.Image image mode=RGB size=32x32 at 0x110A23CB0>, 6)
    train_dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transforms.ToTensor()
    )
    img, label = train_dataset[0]
    print(img.shape, img)
    writer = SummaryWriter("logs")
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    step = 0
    for batch in dataloader:
        images, labels = batch
        writer.add_images("Batch Images2", images, step)
        print(step)
        step += 1
    writer.close()


if __name__ == "__main__":
    main()
