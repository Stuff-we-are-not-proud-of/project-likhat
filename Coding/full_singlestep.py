from single_convolution_krishna import single_convolution_layer, single_convolution_layer_from_matrix, conv_backward
from max_pooling_krishna import max_pool, maxpool_backward
from forward_prop_layers import flatten, densen, dropout
from backward_prop_layers import dense_backward, dropout_backward, flatten_backward, binary_crossentropy_gradient

image_path = r"D:\Project\Coding\not_B_1.ppm"
caches = []

singly_convolved_matrix, cache1 = single_convolution_layer(image_path, filter_size=(3, 3), stride=1, num_filters=32, padding=1)
caches.append(cache1)
singly_maxxed_matrix, cache_max1 = max_pool(singly_convolved_matrix, size=(2, 2), stride=2)
caches.append(cache_max1)

doubly_convolved_matrix, cache2 = single_convolution_layer_from_matrix(singly_maxxed_matrix, filter_size=(3, 3), stride=1, num_filters=64, padding=1)
caches.append(cache2)
doubly_maxxed_matrix, cache_max2 = max_pool(doubly_convolved_matrix, size=(2, 2), stride=2)
caches.append(cache_max2)

triply_convolved_matrix, cache3 = single_convolution_layer_from_matrix(doubly_maxxed_matrix, filter_size=(3, 3), stride=1, num_filters=128, padding=1)
caches.append(cache3)
triply_maxxed_matrix, cache_max3 = max_pool(triply_convolved_matrix, size=(2, 2), stride=2)
caches.append(cache_max3)

flattened_matrix, cache_flatten = flatten(triply_maxxed_matrix)
caches.append(cache_flatten)

singly_densed_matrix, cache_dense1 = densen(flattened_matrix, 128, activation="ReLU")
caches.append(cache_dense1)

dropout_matrix, cache_dropout = dropout(singly_densed_matrix, dropout_rate=0.5, training=True)
caches.append(cache_dropout)

doubly_densed_matrix, cache_dense2 = densen(dropout_matrix, 1, activation="sigmoid")
caches.append(cache_dense2)

print(doubly_densed_matrix)

gradients = {}
y_true = 1
dA = binary_crossentropy_gradient(y_true, doubly_densed_matrix)

dA, dW, db = dense_backward(dA, caches[9], 'sigmoid')
gradients['dense2_dW'] = dW
gradients['dense2_db'] = db

dA = dropout_backward(dA, caches[8])

dA, dW, db = dense_backward(dA, caches[7], 'relu')
gradients['dense1_dW'] = dW
gradients['dense1_db'] = db

dA = flatten_backward(dA, caches[6])

dA = maxpool_backward(dA, caches[5])

dA, dW, db = conv_backward(dA, caches[4])
gradients['conv3_dW'] = dW
gradients['conv3_db'] = db

dA = maxpool_backward(dA, caches[3])

dA, dW, db = conv_backward(dA, caches[2])
gradients['conv2_dW'] = dW
gradients['conv2_db'] = db

dA = maxpool_backward(dA, caches[1])

dA, dW, db = conv_backward(dA, caches[0])
gradients['conv1_dW'] = dW
gradients['conv1_db'] = db

