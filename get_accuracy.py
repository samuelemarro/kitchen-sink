import click
import torch
import torch.nn as nn
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


@click.command()
@click.argument('model_path')
@click.option('--coeff', type=float, default=1)
def main(model_path, coeff):
    device = 'cuda'
    batch_size = 128

    test_dataset = data.parse_dataset('cifar10', 'std:test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    model = models.mini_model(coeff=coeff)
    
    model.load_state_dict(torch.load(model_path))
    acc = accuracy(model, test_dataloader, device)
    print('Accuracy: ', acc * 100.0, '%')


if __name__ == '__main__':
    main()