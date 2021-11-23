import click
import numpy as np
import torch

import data
import models
import parsing
import utils

@click.command()
@click.argument('coefficient')
@click.argument('attack')
@click.argument('count', type=int)
@click.argument('state_dict_path')
def main(coefficient, attack, count, state_dict_path):
    model = models.mini_model(coeff=float(coefficient))
    with open(state_dict_path) as f:
        model.load_state_dict(torch.load(f))

    device = 'cuda'
    batch_size = 50
    test_dataset = data.parse_dataset('cifar10', 'std:test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset[:count], batch_size, shuffle=False)

    attack_config = utils.read_attack_config_file('default_attack_configuration.cfg')

    attack = parsing.parse_attack(attack, 'cifar10', 'linf', 'standard', model, attack_config, device)

    distances = []

    for images, labels in test_dataloader:
        adversarials = attack(images, y=labels)

        distances.append(torch.max(torch.abs(images - adversarials), dim=(1, 2, 3)))

    distances = torch.stack(distances)

    distances = distances.cpu().numpy()

    print('Median: ', np.median(distances))
    print('Average: ', np.average(distances))
