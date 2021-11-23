import numpy as np
import torch
from torch import nn

from tqdm import tqdm

import data
import models
import utils


def accuracy(model, loader, device):
    correct_count = 0
    total_count = 0

    model.to(device)

    for images, true_labels in tqdm(loader, desc='Accuracy Test'):
        total_count += len(images)
        images = images.to(device)
        true_labels = true_labels.to(device)

        predicted_labels = utils.get_labels(model, images).detach()

        correct = torch.eq(predicted_labels, true_labels)
        correct_count += len(torch.nonzero(correct))

    return correct_count / total_count

model = models.mini_model()

model.load_state_dict(torch.load('trained-models/test/run_1/kitchen-sink.pth'))

device = 'cuda'
batch_size = 64
loss_function = nn.CrossEntropyLoss()

dataset = data.parse_dataset('cifar10', 'std:train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
test_dataset = data.parse_dataset('cifar10', 'std:test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

mask_set = utils.load_zip('trained-models/test/run_1/mask_set.zip')
lr_tensors = utils.load_zip('trained-models/test/run_1/lr_tensors.zip')

n_lr = len(torch.unique(lr_tensors[0]))
n_mask = len(mask_set[0])

model.to(device)

masked_models = [models.mini_model().to(device).eval() for _ in range(n_lr * n_mask)]

assert len(mask_set) == len(lr_tensors) == len(list(model.parameters()))

for parameter_index, (mask_variants, lr_tensor, parameter) in enumerate(zip(mask_set, lr_tensors, model.parameters())):
    for i, lr in enumerate(torch.unique(lr_tensor)):
        for j, mask in enumerate(mask_variants):
            index = i * n_mask + j
            lr_mask = lr_tensor.eq(lr)
            true_mask = mask & lr_mask

            list(masked_models[index].parameters())[parameter_index].data = parameter * (~true_mask).float()

train_accuracies = []
test_accuracies = []

for model_index, masked_model in enumerate(masked_models):
    """train_accuracy = accuracy(masked_model, dataloader, device)

    print(f'Model {model_index} masked train accuracy: ', train_accuracy * 100.0, '%')
    train_accuracies.append(train_accuracy)"""

    test_accuracy = accuracy(masked_model, test_dataloader, device)

    print(f'Model {model_index} masked test accuracy: ', test_accuracy * 100.0, '%')
    test_accuracies.append(test_accuracy)

# print('Best train: ', np.argsort(train_accuracies))
print('Best test: ', np.argsort(test_accuracies))