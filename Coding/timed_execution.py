import time
import numpy as np
from max_pooling_krishna import max_pool
from single_convolution_krishna import single_convolution_layer

image_path = r"C:\Users\Krishna Gera\Desktop\Project Likhat\Coding\Arhan_c_aug_1.ppm"

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

timed_convolution = timed_execution(single_convolution_layer)
timed_maxpool = timed_execution(max_pool)

return_of_convolve = timed_convolution(image_path, (3,3), 1)
return_of_pool = timed_maxpool(return_of_convolve, 2, 2)

print(f"Convolution Shape : {return_of_convolve.shape[0]}, {return_of_convolve.shape[1]}, {return_of_convolve.shape[2]}")
print(f"Pool Shape : {return_of_pool.shape[0]}, {return_of_pool.shape[1]}, {return_of_pool.shape[2]}")