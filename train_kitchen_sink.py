import torch
import torch.nn as nn
import torch.optim

import itertools

from tqdm import tqdm

import data
import models
import utils
import numpy as np

import pathlib

import optimizers

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

def basic_train(model, dataloader, parameter_set, optimizer, loss_function, device):
    model.train()
    model.to(device)

    max_epochs = parameter_set['epochs']

    iterator = tqdm(range(max_epochs), desc='Training')

    for _ in iterator:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            loss = loss_function(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

def ks_train(model, dataloader, parameter_sets, optimizers, loss_function, device):
    model.train()
    model.to(device)

    max_epochs = max([parameter_set['epochs'] for parameter_set in parameter_sets])

    iterator = tqdm(range(max_epochs), desc='Training')

    for i in iterator:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            loss = loss_function(y_pred, labels)

            optimizers[0].zero_grad()
            loss.backward()

            for parameter_set, optimizer in zip(parameter_sets, optimizers):
                if i < parameter_set['epochs']:
                    optimizer.step()

def basic_ks_train(model, dataloader, epoch_set, optimizer, loss_function, device):
    model.train()
    model.to(device)

    max_epochs = max(epoch_set)

    iterator = tqdm(range(max_epochs), desc='Training')

    for i in iterator:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            loss = loss_function(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()

            active_masks = [i < epoch for epoch in epoch_set]

            optimizer.step(active_masks)

def get_basic_kitchen_sink_optimizer(model, lrs, epochs, device):
    parameters = list(model.parameters())

    lr_tensors = []
    mask_set = []

    for parameter in parameters:
        lr_tensor = torch.zeros_like(parameter, device=device)

        selections = torch.randint(0, len(lrs), lr_tensor.shape, device=device)

        for i, lr in enumerate(lrs):
            same_selection = torch.eq(selections, i)

            lr_tensor += lr * same_selection.float()

        assert not torch.eq(lr_tensor, 0).any()
        lr_tensors.append(lr_tensor)

        selections = torch.randint(0, len(epochs), parameter.shape, device=device)

        mask_variants = [torch.eq(selections, i) for i in range(len(epochs))]
        mask_set.append(mask_variants)
    
    return optimizers.AdamLRMasked(parameters, lr_tensors, mask_set), lr_tensors, mask_set

def main():
    for i in range(3, 4): # Temp
        device = 'cuda'
        batch_size = 128
        loss_function = nn.CrossEntropyLoss()

        dataset = data.parse_dataset('cifar10', 'std:train')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

        name = f'ks/1.1/{i}'
        coeff = 1.1
        base_folder = pathlib.Path('trained-models') / name
        base_folder.mkdir(parents=True, exist_ok=True)

        test_dataset = data.parse_dataset('cifar10', 'std:test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

        lrs = [1e-3, 1e-4, 1e-5]
        epochs = [100, 200, 300]

        model = models.mini_model(coeff=coeff)
        optimizer, lr_tensors, mask_set = get_basic_kitchen_sink_optimizer(model, lrs, epochs, device)
        basic_ks_train(model, dataloader, epochs, optimizer, loss_function, device)

        torch.save(model.state_dict(), base_folder / 'kitchen-sink.pth')
        acc = accuracy(model, test_dataloader, device)
        print('KS Accuracy: ', acc * 100.0, '%')

        utils.save_zip(lr_tensors, base_folder / 'lr_tensors.zip')
        utils.save_zip(mask_set, base_folder / 'mask_set.zip')
            

if __name__ == '__main__':
    main()