import numpy as np

def densen(input_array, output_size, activation="ReLU"):
    input_size = len(input_array)
    limit = np.sqrt(6 / input_size)
    weights = np.random.uniform(-limit, limit, size=(input_size, output_size))
    biases = np.zeros(output_size)

    multiplied_matrix = input_array @ weights
    added_to_multiplied = multiplied_matrix + biases
    output = added_to_multiplied
    if(activation == "ReLU"):
        output = np.maximum(0, output)
    elif(activation == "sigmoid"):
        output = 1 / (1 + np.exp(-output))
    
    return output