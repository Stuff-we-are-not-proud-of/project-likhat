from conv_forward_shivansh import initialize_conv_layer, conv_forward, conv_forward_from_matrix
from max_pooling_shivansh import max_pooling
from flatten import flatten
from dropout import dropout
from dense import densen

# Path to your input image
image_path = r"D:\Project\Coding\not_B_1.ppm"

# 1st conv + pool
filters1, biases1 = initialize_conv_layer(32, 3, 3)
conv1 = conv_forward(image_path, filters1, biases1, stride=1, padding=1)
pool1 = max_pooling(conv1, size=2, stride=2)

# 2nd conv + pool
filters2, biases2 = initialize_conv_layer(64, 3, pool1.shape[-1])
conv2 = conv_forward_from_matrix(pool1, filters2, biases2, stride=1, padding=0)
pool2 = max_pooling(conv2, size=2, stride=2)

# 3rd conv + pool
filters3, biases3 = initialize_conv_layer(128, 3, pool2.shape[-1])
conv3 = conv_forward_from_matrix(pool2, filters3, biases3, stride=1, padding=0)
pool3 = max_pooling(conv3, size=2, stride=2)

# Flatten
flat = flatten(pool3)

# Dense + Dropout + Dense
dense1 = densen(flat, 128, activation="ReLU")
dropout_out, _ = dropout(dense1, dropout_rate=0.5, training=True)
output = densen(dropout_out, 1, activation="sigmoid")

print("Final Output:", output)
