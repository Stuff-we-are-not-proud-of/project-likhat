import numpy as np

def max_pool(image_matrix, size=(2, 2), stride=2):
    input_height, input_width, input_channels = image_matrix.shape
    output_height = ((input_height - size[0]) // stride) + 1
    output_width = ((input_width - size[1]) // stride) + 1
    output_matrix = np.zeros((output_height, output_width, input_channels))
    
    for i in range(output_height):
        for j in range(output_width):
            for k in range(input_channels):
                row_start = i * stride
                col_start = j * stride
                sub = image_matrix[row_start:row_start + size[0], col_start:col_start + size[1], k]
                if sub.size > 0:
                    output_matrix[i, j, k] = np.max(sub)
                else:
                    print(f"Warning: Empty subarray at ({i}, {j}, {k})")
    return output_matrix