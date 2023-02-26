# predict.py

import mxnet as mx
import numpy as np
import cv2, os
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# Load the network parameters
sym, arg_params, aux_params = mx.model.load_checkpoint('image-classification', 5)

# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,28,28))])
mod.set_params(arg_params, aux_params, allow_missing=True)

def predict(filename, mod):
    img = cv2.imread(filename)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (28, 28))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    a = np.argsort(prob)[::-1]
    return (a[0], prob[a[0]])

# Code to predict on a local file
def make_prediction(filename):
    return predict(filename, mod)

