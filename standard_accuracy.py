import os
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

import data
import models
import utils

VERBOSE = False

def accuracy(model, loader, device):
    correct_count = 0
    total_count = 0

    model.to(device)

    for images, true_labels in (tqdm(loader, desc='Accuracy Test') if VERBOSE else loader):
        total_count += len(images)
        images = images.to(device)
        true_labels = true_labels.to(device)

        predicted_labels = utils.get_labels(model, images).detach()

        correct = torch.eq(predicted_labels, true_labels)
        correct_count += len(torch.nonzero(correct))

    return correct_count / total_count


@click.command()
@click.argument('coefficient', type=str)
def main(coefficient):
    device = 'cuda'
    batch_size = 128
    test_dataset = data.parse_dataset('cifar10', 'std:test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)


    base_folder = Path(f'trained-models/standard/{coefficient}')

    accuracies = [list() for _ in range(9)]

    for subdirectory in range(1, 6):
        run_folder = base_folder / str(subdirectory)
        for i in range(len(accuracies)):
            model_path = run_folder / f'{i}.pth'
            model = models.mini_model(coeff=float(coefficient))
        
            if VERBOSE:
                print(model_path)
            model.load_state_dict(torch.load(model_path))
            acc = accuracy(model, test_dataloader, device)
            accuracies[i].append(acc)
    
    utils.save_zip(np.array(accuracies), f'accuracies/standard_1/{coefficient}.zip')

    print(accuracies)
    average_accuracies = []

    for i, accuracy_set in enumerate(accuracies):
        average_accuracy = np.mean(accuracy_set)

        average_accuracies.append(average_accuracy)

        if VERBOSE:
            import time
            time.sleep(0.5)

            print()
            print(f'Model {i}:')
            print('Average accuracy: ', average_accuracy * 100.0, '%')
            print()
    
    print('Leaderboard:')

    for index in np.argsort(-np.array(average_accuracies)):
        print(f'{index} ({average_accuracies[index]})')

if __name__ == '__main__':
    main()