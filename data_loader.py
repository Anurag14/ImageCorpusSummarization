import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

transform = transforms.Compose(
    [transforms.Resize(64),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_loader(config):
    if config.dataset == 'CIFAR10' and config.mode == 'train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=False, num_workers=2)
    if config.dataset == 'CIFAR10' and config.mode == 'test':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)
    if config.dataset == 'CIFAR100' and config.mode == 'train':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=False, num_workers=2)
    if config.dataset == 'CIFAR100' and config.mode == 'test':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)
    return dataloader
