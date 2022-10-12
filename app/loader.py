# 프로젝트
from config import CONFIG

# 서드파티
import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10:
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = CONFIG.batch_size
    train_data = torchvision.datasets.CIFAR10('./data/CIFAR10/train',train=True, download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.CIFAR10('./data/CIFAR10/test', train=False, download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)