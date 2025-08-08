from ppm_to_matrix_shivansh import extract_rgb_matrix
from conv_forward_shivansh import conv_forward_from_matrix,conv_forward
from max_pooling_shivansh import max_pooling
from flatten_shivansh import flatten
from dense_shivansh import densen
from dropout_shivansh import dropout

# ----------------------------
# Load input image
# ----------------------------
image_path = r"D:\Project\Coding\not_B_1.ppm" 

# ----------------------------
# Conv Layer 1 + Pool
# ----------------------------
conv1_out, cache_conv1 = conv_forward(image_path, num_filters=32, filter_size=3, stride=1, padding=1)
pool1_out, cache_pool1 = max_pooling(conv1_out, size=2, stride=2)


conv2_out, cache_conv2 = conv_forward_from_matrix(pool1_out, num_filters=64, filter_size=3, stride=1, padding=0)
pool2_out, cache_pool2 = max_pooling(conv2_out, size=2, stride=2)


conv3_out, cache_conv3 = conv_forward_from_matrix(pool2_out, num_filters=128, filter_size=3, stride=1, padding=0)
pool3_out, cache_pool3 = max_pooling(conv3_out, size=2, stride=2)



flat_out, cache_flat = flatten(pool3_out)

dense1_out, cache_dense1 = densen(flat_out, output_size=128, activation="ReLU")

dropout_out, cache_dropout = dropout(dense1_out, dropout_rate=0.5, training=True)
dense2_out, cache_dense2 = densen(dropout_out, output_size=1, activation="sigmoid")

print("Final Output:", dense2_out)

caches = {
    "conv1": cache_conv1, "pool1": cache_pool1,
    "conv2": cache_conv2, "pool2": cache_pool2,
    "conv3": cache_conv3, "pool3": cache_pool3,
    "flat": cache_flat,
    "dense1": cache_dense1, "dropout": cache_dropout,
    "dense2": cache_dense2
}
