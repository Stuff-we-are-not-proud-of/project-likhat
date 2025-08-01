import time
import numpy as np
from max_pooling_shivansh import max_pooling
from conv_forward_shivansh import conv_forward
from conv_forward_shivansh import initialize_conv_layer

image_path = r"D:\Project\Coding\Shivansh_A_aug_1.ppm"

def timed_execution(func): 
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Execution Time of {func.__name__}: {execution_time:.6f} seconds")
        print(f"Execution Time of {func.__name__}: {execution_time * 1000:.6f} milliseconds")
        return result
    return wrapper

timed_convolution = timed_execution(conv_forward)
timed_maxpool = timed_execution(max_pooling)
# Initialize filters & biases
filters, biases = initialize_conv_layer(8, 3, 3)

# Run convolution
return_of_convolve = timed_convolution(image_path, filters, biases, stride=1, padding=1)
return_of_pool = timed_maxpool(return_of_convolve, 2, 2)

print(f"Convolution Shape : {return_of_convolve.shape[0]}, {return_of_convolve.shape[1]}, {return_of_convolve.shape[2]}")
print(f"Pool Shape : {return_of_pool.shape[0]}, {return_of_pool.shape[1]}, {return_of_pool.shape[2]}")