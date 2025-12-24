import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

def main():
    print("Hello from pytorch-learning!")
    
    # 1. torchvision中Transforms的基本使用演示
    print("\n1. torchvision中Transforms的基本使用演示")
    
    # 加载图片
    img_path = "demo1.png"
    if not os.path.exists(img_path):
        print(f"图片文件 {img_path} 不存在")
        return
    
    img = Image.open(img_path)
    # 将RGBA图像转换为RGB图像
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    print(f"原始图片类型: {type(img)}, 模式: {img.mode}, 尺寸: {img.size}")
    
    # 创建各种Transforms
    transform_list = [
        ("ToTensor", transforms.ToTensor()),
        ("Resize", transforms.Resize((256, 256))),
        ("CenterCrop", transforms.CenterCrop(200)),
        ("RandomCrop", transforms.RandomCrop(200)),
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1)),
        ("RandomVerticalFlip", transforms.RandomVerticalFlip(p=1)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)),
        ("Normalize", transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
    ]
    
    # 2. 结合TensorBoard演示图片变换过程
    print("\n2. 结合TensorBoard演示图片变换过程")
    
    # 创建SummaryWriter
    writer = SummaryWriter("logs")
    
    # 写入原始图片
    img_tensor = transforms.ToTensor()(img)
    writer.add_image("Original Image", img_tensor, 0)
    
    # 应用各种变换并写入TensorBoard
    for i, (transform_name, transform) in enumerate(transform_list):
        if transform_name == "Normalize":
            # Normalize需要先转换为Tensor
            transformed_img = transform(img)
        else:
            transformed_img = transform(img)
            # 如果不是Tensor，转换为Tensor以便写入TensorBoard
            if not isinstance(transformed_img, torch.Tensor):
                transformed_img = transforms.ToTensor()(transformed_img)
        
        writer.add_image(f"Original Image", transformed_img, i+1)
        print(f"应用 {transform_name} 变换后的图片形状: {transformed_img.shape}")
    
    # 关闭SummaryWriter
    writer.close()
    
    print("\n3. 总结")
    print("- torchvision是PyTorch的计算机视觉库，提供数据集、模型、变换等功能")
    print("- Transforms用于对图像进行各种预处理和增强操作")
    print("- TensorBoard可以可视化图像变换过程")
    print("\n运行 'tensorboard --logdir=logs' 可以查看可视化结果")


if __name__ == "__main__":
    main()
