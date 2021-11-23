import os
from pathlib import Path

import click
import numpy as np
import torch
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
@click.argument('coefficient')
def main(coefficient):
    device = 'cuda'
    batch_size = 128
    test_dataset = data.parse_dataset('cifar10', 'std:test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)


    base_folder = Path('trained-models/ks')
    coefficients = [coefficient]

    accuracies = [list() for _ in range(len(coefficients))]

    for index, coefficient in enumerate(coefficients):
        run_folder = base_folder / coefficient
        for i in range(5):
            model_path = run_folder / str(i + 1) / 'kitchen-sink.pth'
            model = models.mini_model(coeff=float(coefficient))
        
            print(model_path)
            model.load_state_dict(torch.load(model_path))
            acc = accuracy(model, test_dataloader, device)
            accuracies[index].append(acc)

    print(accuracies)

    for coefficient, accuracy_set in zip(coefficients, accuracies):
        average_accuracy = np.mean(accuracy_set)

        import time
        time.sleep(0.5)

        print(f'Model {coefficient}:')
        print('Average accuracy: ', average_accuracy * 100.0, '%')

if __name__ == '__main__':
    main()