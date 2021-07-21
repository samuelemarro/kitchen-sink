import torch
from torch._C import device
import torch.nn as nn
import torch.optim
from torch.optim import optimizer
import optimizers as optim
import torchvision

import itertools

from tqdm import tqdm

import data
import models
import utils
import numpy as np

import pathlib

# Batch size: aggiorni i pesi solo dopo un tot
# Coefficienti?
# Epochs: blocchi alcuni pesi dopo un tot
# Immagini diverse: ti tocca fare un forward/backward diverso? Però solo per immagini che sono diverse
# Coefficienti di loss aggiunte linearmente: calcoli tutte le loss indipendentemente, sommi e applichi in maniera specifica
# Parametri specifici di loss: temo che si debba ricalcolare
# Comportamenti diversi della rete (es. dropout diversi): fuck it, tutto insieme

# Tecnica 1: universi paralleli
# Questa è la tecnica più universale, ma anche la più inefficiente
# Si creano tanti universi, uno per ogni combinazione di iperparametri
# Ogni peso ha un universo associato
# Per ogni sample, esistono le versioni di tutti gli universi


# Eseguire l'update
# Per alcune tecniche (es. universi paralleli), è possibile avere situazioni
# in cui devi calcolare weight updates diversi per pesi diversi. Ci sono due approcci possibili:
# - Calcola tutti i weight updates e applicali tutti insieme
#   - Forse si può fare salvando esclusivamente i grad di ogni update
# - Calcola in ordine casuale i weight updates e applicali insieme
#   - Pro: hai il vantaggio di velocità delle minibatch
#   - Contro: hai lo svantaggio di instabilità delle minibatch


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

def get_optimizer(model_parameters, parameter_set, mask=None):
    if parameter_set['optimizer_type'] == 'adam':
        if mask is None:
            return torch.optim.Adam(model_parameters, lr=parameter_set['lr'])
        else:
            return optim.AdamMasked(model_parameters, mask, lr=parameter_set['lr'])
    elif parameter_set['optimizer_type'] == 'sgd':
        if mask is None:
            return torch.optim.SGD(model_parameters, lr=parameter_set['sgd'])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError



def get_kitchen_sink_optimizers(parameter_sets, model, device):
    parameters = list(model.parameters())

    mask_sets = [[None] * len(parameters)] * len(parameter_sets)

    for j, parameter in enumerate(model.parameters()):
        indices = np.arange(np.prod(parameter.shape))
        np.random.shuffle(indices)
        index_partitions = np.array_split(indices, len(parameter_sets), axis=0)

        for i, index_partition in enumerate(index_partitions):
            index_partition = np.unravel_index(index_partition, shape=parameter.shape)
            mask = torch.zeros_like(parameter, device=device, dtype=torch.bool)
            mask[index_partition] = True
            mask_sets[i][j] = mask

    for k in range(len(parameters)):
        mask_sum = torch.zeros_like(mask_sets[0][k], dtype=torch.float)
        for mask_set in mask_sets:
            mask_sum += mask_set[k].float()
            print(mask_set[k].float().sum())
            print(np.prod(mask_set[k].shape))
            #assert False
        print(mask_sum)
        assert torch.all(torch.eq(mask_sum, 1))

    optimizers = []

    for parameter_set, mask_set in zip(parameter_sets, mask_sets):
        optimizers.append(get_optimizer(parameters, parameter_set, mask=mask_set))

    return optimizers, mask_sets


def main():
    device = 'cuda'
    batch_size = 32
    loss_function = nn.CrossEntropyLoss()

    dataset = data.parse_dataset('cifar10', 'std:train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    name = 'test'
    base_folder = pathlib.Path('trained-models') / name
    base_folder.mkdir(parents=True, exist_ok=True)
    kitchen_sink = True

    test_dataset = data.parse_dataset('cifar10', 'std:test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # parameter_sets = create_basic_set(optimizer_type=['adam'], lr=[1e-3, 1e-4, 1e-5], epochs=[100, 200, 300])

    parameter_sets = create_basic_set(optimizer_type=['adam'], lr=[1e-3, 1e-4, 1e-5], epochs=[100, 200, 300])

    if kitchen_sink:
        model = models.mini_model()
        optimizers, mask_sets = get_kitchen_sink_optimizers(parameter_sets, model, device)
        assert len(parameter_sets) == len(optimizers)
        ks_train(model, dataloader, parameter_sets, optimizers, loss_function, device)

        torch.save(model.state_dict(), base_folder / 'kitchen-sink.pth')
        utils.save_zip(mask_sets, base_folder / 'mask_sets.zip')
        acc = accuracy(model, test_dataloader, device)
        print('KS Accuracy: ', acc * 100.0, '%')
    else:
        for i, parameter_set in enumerate(parameter_sets):
            instance_path = base_folder / f'{i}.pth'
            model = models.mini_model()
            optimizer = get_optimizer(model.parameters(), parameter_set)
            basic_train(model, dataloader, parameter_set, optimizer, loss_function, device)

            torch.save(model.state_dict(), instance_path)
            acc = accuracy(model, test_dataloader, device)
            print(parameter_set)
            print(f'{i} Accuracy: ', acc * 100.0, '%')
            

if __name__ == '__main__':
    main()