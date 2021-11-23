import torch
import torch.nn as nn
import torch.optim

import itertools

from tqdm import tqdm

import data
import models
import numpy as np

import pathlib

import utils

import click

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

def create_basic_set(**parameters):
    parameter_pairs = []
    for key, value_list in parameters.items():
        parameter_pairs.append([(key, val) for val in value_list])
    
    parameter_sets = []
    
    for product in itertools.product(*parameter_pairs):
        parameter_set = {}
        for key, value in product:
            parameter_set[key] = value

        parameter_sets.append(parameter_set)
    
    return parameter_sets

def get_optimizer(model_parameters, parameter_set):
    if parameter_set['optimizer_type'] == 'adam':
        return torch.optim.Adam(model_parameters, lr=parameter_set['lr'])
    elif parameter_set['optimizer_type'] == 'sgd':
        return torch.optim.SGD(model_parameters, lr=parameter_set['lr'])
    else:
        raise NotImplementedError

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

@click.command()
@click.argument('coefficient', type=str)
@click.argument('run', type=int)
def main(coefficient, run):
    device = 'cuda'
    batch_size = 128
    loss_function = nn.CrossEntropyLoss()

    dataset = data.parse_dataset('cifar10', 'std:train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    base_folder = pathlib.Path('trained-models/standard') / coefficient / str(run)
    base_folder.mkdir(parents=True, exist_ok=True)

    test_dataset = data.parse_dataset('cifar10', 'std:test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # parameter_sets = create_basic_set(optimizer_type=['adam'], lr=[1e-3, 1e-4, 1e-5], epochs=[100, 200, 300])

    lrs = [1e-3, 1e-4, 1e-5]
    epochs = [100, 200, 300]

    parameter_sets = create_basic_set(optimizer_type=['adam'], lr=lrs, epochs=epochs)

    for i, parameter_set in enumerate(parameter_sets):
            instance_path = base_folder / f'{i}.pth'
            if not instance_path.exists():
                model = models.mini_model(coeff=float(coefficient))

                optimizer = get_optimizer(model.parameters(), parameter_set)
                basic_train(model, dataloader, parameter_set, optimizer, loss_function, device)

                torch.save(model.state_dict(), instance_path)
                acc = accuracy(model, test_dataloader, device)
                print(parameter_set)
                print(f'{i} Accuracy: ', acc * 100.0, '%')


if __name__ == '__main__':
    main()