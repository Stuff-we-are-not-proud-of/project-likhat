import numpy as np

def dropout(input_array, dropout_rate=0.5, training=True):
    if not training:
        return input_array  # No dropout during inference
    
    # Create a mask with keep probability (1 - dropout_rate)
    keep_prob = 1 - dropout_rate
    mask = np.random.binomial(1, keep_prob, size=input_array.shape) / keep_prob  # Scale to maintain expected value
    
    # Apply mask
    masked_output = input_array * mask
    
    return masked_output  # Return only the masked array, shape same as input_array