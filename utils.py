import gzip
import json
import pathlib
import pickle

import torch


def get_labels(model, images):
    model_device = next(model.parameters()).device
    outputs = model(images.to(model_device))
    assert len(outputs) == len(images)
    assert len(outputs.shape) == 2

    return torch.argmax(outputs, axis=1).to(images.device)


def save_zip(obj, path, protocol=0):
    """
    Saves a compressed object to disk.
    """
    # Create the folder, if necessary
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    file = gzip.GzipFile(path, 'wb')
    pickled = pickle.dumps(obj, protocol)
    file.write(pickled)
    file.close()

def load_zip(path):
    """
    Loads a compressed object from disk.
    """
    file = gzip.GzipFile(path, 'rb')
    buffer = b''
    while True:
        data = file.read()
        if data == b'':
            break
        buffer += data
    obj = pickle.loads(buffer)
    file.close()
    return obj

class AttackConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def get_arguments(self, attack_name, domain, p, attack_type):
        kwargs = {}

        def load_kwargs(new_kwargs):
            for key, value in new_kwargs.items():
                kwargs[key] = value

        def loop_across_dict(current_dict, selectors):
            if 'params' in current_dict:
                load_kwargs(current_dict['params'])

            if len(selectors) == 0:
                return

            general_selector, specific_selector = selectors[0]

            if general_selector in current_dict and specific_selector in current_dict:
                raise RuntimeError('Both selectors available: cannot choose.')

            if specific_selector in current_dict:
                loop_across_dict(current_dict[specific_selector], selectors[1:])
            elif general_selector in current_dict:
                assert len(current_dict.keys()) <= 2
                loop_across_dict(current_dict[general_selector], selectors[1:])

        # The specific value overrides the general one, from outermost to innermost
        loop_across_dict(self.config_dict,
                         [
                             ('all_attacks', attack_name),
                             ('all_domains', domain),
                             ('all_distances', p),
                             ('all_types', attack_type)
                         ]
                         )

        return kwargs

def read_attack_config_file(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)

    return AttackConfig(config_dict)