import torch
import torch.functional as F
import torchvision
from torchvision import transforms

def parse_dataset(domain, dataset, extra_transforms=None):
    if extra_transforms is None:
        extra_transforms = []
    matched_dataset = None
    tensor_transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose(
        extra_transforms + [tensor_transform])

    if domain == 'cifar10':
        if dataset == 'std:train':
            matched_dataset = torchvision.datasets.CIFAR10(
                './data/cifar10', train=True, download=True, transform=transform)
        elif dataset == 'std:test':
            matched_dataset = torchvision.datasets.CIFAR10(
                './data/cifar10', train=False, download=True, transform=transform)
    elif domain == 'mnist':
        if dataset == 'std:train':
            matched_dataset = torchvision.datasets.MNIST(
                './data/mnist', train=True, download=True, transform=transform)
        elif dataset == 'std:test':
            matched_dataset = torchvision.datasets.MNIST(
                './data/mnist', train=False, download=True, transform=transform)

    return matched_dataset

class KSTransform(torch.nn.Module):
    pass

class KSCompose(KSTransform):
    def __init__(self, transform_sets):
        super().__init__()

        self.transform_sets = transform_sets
    
    def forward(self, img):
        pass

class KSRandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip, KSTransform):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), False
        return img, True
    