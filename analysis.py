import numpy as np
import torch

import models
import utils


model = models.mini_model()

model.load_state_dict(torch.load('trained-models/test/kitchen-sink.pth'))

mask_sets = utils.load_zip('trained-models/test/mask_sets.zip')

norms = []

for mask_set in mask_sets:
    norm = 0
    print('Set')
    for mask, parameters in zip(mask_set, model.parameters()):
        print('Mask: ', mask[(0,) * len(mask.shape)])
        norm += torch.sum(parameters[mask] ** 2)
    
    norms.append(norm.detach().cpu().numpy())

norms = np.array(norms)

print(norms)
print('Best: ', np.argmax(norms))