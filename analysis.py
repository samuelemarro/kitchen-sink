from pathlib import Path
import click

import numpy as np
import torch
from torch import nn

import data
import models
import utils

def top_k(values, k):
    return values.argsort()[-k:][::-1]

def run_one(coefficient, folder, analysis):
    folder = Path(folder)
    model_path = folder / 'kitchen-sink.pth'
    mask_set_path = folder / 'mask_set.zip'
    lr_tensors_path = folder / 'lr_tensors.zip'

    model = models.mini_model(coeff=coefficient)

    model.load_state_dict(torch.load(model_path))

    device = 'cuda'
    batch_size = 256
    loss_function = nn.CrossEntropyLoss()

    # dataset = data.parse_dataset('cifar10', 'std:train')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    test_dataset = data.parse_dataset('cifar10', 'std:test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    mask_set = utils.load_zip(mask_set_path)
    lr_tensors = utils.load_zip(lr_tensors_path)

    n_lr = len(torch.unique(lr_tensors[0]))
    n_mask = len(mask_set[0])

    model.to(device)

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        y_pred = model(images)
        loss = loss_function(y_pred, labels)
        loss.backward()

    scores = [0] * n_lr * n_mask

    for mask_variants, lr_tensor, parameter in zip(mask_set, lr_tensors, model.parameters()):
        for i, lr in enumerate(torch.unique(lr_tensor)):
            all_weights = []
            for j, mask in enumerate(mask_variants):
                index = i * n_mask + j
                lr_mask = lr_tensor.eq(lr)
                true_mask = mask & lr_mask

                if analysis == 'grad_abs':
                    grad_sum = torch.sum(torch.abs(parameter.grad[true_mask])).cpu()
                    scores[index] += grad_sum
                elif analysis == 'grad_squared':
                    grad_sum = torch.sum(parameter.grad[true_mask] ** 2).cpu()
                    scores[index] += grad_sum
                elif analysis == 'abs':
                    scores[index] += torch.sum(torch.abs(parameter[true_mask])).cpu()
                elif analysis == 'squared':
                    scores[index] += torch.sum(parameter[true_mask] ** 2).cpu()
                elif analysis.startswith('pruning'):
                    all_weights.append(parameter[true_mask].flatten())
                elif analysis == 'pruning_grad':
                    all_weights.append(parameter.grad[true_mask].flatten())
                else:
                    raise NotImplementedError
            
            if analysis.startswith('pruning'):
                all_weights = torch.cat(all_weights)
                threshold = torch.quantile(all_weights, 0.5)

                for j, mask in enumerate(mask_variants):
                    index = i * n_mask + j
                    lr_mask = lr_tensor.eq(lr)
                    true_mask = mask & lr_mask

                    if analysis == 'pruning':
                        interesting_value = parameter
                    elif analysis == 'pruning_grad':
                        interesting_value = parameter.grad
                    scores[index] += torch.count_nonzero(interesting_value[true_mask] >= threshold)

    scores = np.array(scores)

    return np.argmax(scores), np.argmin(scores) #(top_k(norms, 2), top_k(-norms, 2))

@click.command()
@click.argument('coefficient', type=str)
@click.argument('analysis', type=str)
def main(coefficient, analysis):
    base_folder = Path('trained-models/ks') / coefficient

    highest_count = [0] * 9
    lowest_count = [0] * 9

    for i in range(5):
        run_folder = base_folder / str(i + 1)
        try:
            highest, lowest = run_one(float(coefficient), run_folder, analysis)
            highest_count[highest] += 1
            lowest_count[lowest] += 1
        except:
            pass
    
    for index in range(9):
        print(f'{index}: Highest {highest_count[index]}, lowest {lowest_count[index]}')


if __name__ == '__main__':
    main()