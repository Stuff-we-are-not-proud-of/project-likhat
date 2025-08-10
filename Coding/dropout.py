import numpy as np

def dropout(input_array, dropout_rate=0.5, training=True):
    if not training:
        return input_array, {}
    keep_prob = 1 - dropout_rate
    mask = np.random.binomial(1, keep_prob, size=input_array.shape) / keep_prob 
    
    masked_output = input_array * mask
    cache = {
        'mask': mask
    }
    return masked_output, cache