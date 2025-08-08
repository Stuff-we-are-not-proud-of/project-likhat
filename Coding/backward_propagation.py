import numpy as np

def binary_crossentropy_loss(y_true, y_pred):
    return - (y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))

def binary_crossentropy_gradient(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-10)  