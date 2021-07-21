import gzip
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