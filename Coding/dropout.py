import numpy as np

def dropout(input_array, dropout_rate=0.5, training=True):

    if training:
        mask = np.random.binomial(1, 1 - dropout_rate, size=input_array.shape)
        
        # Apply mask and scale activations
        output_array = (input_array * mask) / (1 - dropout_rate)
        return output_array, mask
    else:
        # No dropout at inference
        return input_array, None
