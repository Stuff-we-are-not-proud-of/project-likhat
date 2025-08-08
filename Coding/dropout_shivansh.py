import numpy as np

def dropout(input_array, dropout_rate=0.5, training=True):
    if not training:
        return input_array, None
    mask = (np.random.rand(*input_array.shape) > dropout_rate).astype(np.float32) / (1.0 - dropout_rate)
    output = input_array * mask
    cache = {
        "mask": mask,
        "dropout_rate": dropout_rate,
        "input_shape": input_array.shape
    }
    return output, cache
