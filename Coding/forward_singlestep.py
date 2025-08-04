import numpy as np
from single_convolution_krishna import single_convolution_layer, single_convolution_layer_from_matrix
from max_pooling_krishna import max_pool
from flatten import flatten
from dropout import dropout
from dense import densen

image_path = r"C:\Users\Krishna Gera\Desktop\Project Likhat\Coding\not_B_1.ppm"
singly_convolved_matrix = single_convolution_layer(image_path, filter_size=(3, 3), stride=1, num_filters=32, padding=1)
singly_maxxed_matrix = max_pool(singly_convolved_matrix, size=(2, 2), stride=2)
doubly_convolved_matrix = single_convolution_layer_from_matrix(singly_maxxed_matrix, filter_size=(3, 3), stride=1, num_filters=64, padding=0)
doubly_maxxed_matrix = max_pool(doubly_convolved_matrix, size=(2, 2), stride=2)
triply_convolved_matrix = single_convolution_layer_from_matrix(doubly_convolved_matrix, filter_size=(3, 3), stride=1, num_filters=128, padding=0)
triply_maxxed_matrix = max_pool(triply_convolved_matrix, size=(2, 2), stride=2)
flattened_matrix = flatten(triply_maxxed_matrix)
singly_densed_matrix = densen(flattened_matrix, 128, activation="ReLU")
dropout_matrix = dropout(singly_densed_matrix, dropout_rate=0.5, training=True)
doubly_densed_matrix = densen(dropout_matrix, 1, activation="sigmoid")

print(doubly_densed_matrix)