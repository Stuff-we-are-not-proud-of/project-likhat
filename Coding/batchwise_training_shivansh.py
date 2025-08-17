import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

from conv_forward_shivansh import conv_forward, conv_forward_from_matrix, initialize_conv_layer
from conv_backward_shivansh import conv_backward
from max_pooling_shivansh import max_pooling
from maxpool_backward_shivansh import max_pool_backward
from forward_prop_layers_shivansh import flatten, densen
from backward_prop_layers_shivansh import (
    dense_backward, dropout_backward, flatten_backward,
    binary_crossentropy_gradient, binary_crossentropy_loss
)

def load_dataset(model_folder):
    a_folder = os.path.join(model_folder, 'A')
    not_a_folder = os.path.join(model_folder, 'NotA')
    paths, labels = [], []

    for f in os.listdir(a_folder):
        if f.lower().endswith('.ppm'):
            paths.append(os.path.join(a_folder, f))
            labels.append(1)

    for f in os.listdir(not_a_folder):
        if f.lower().endswith('.ppm'):
            paths.append(os.path.join(not_a_folder, f))
            labels.append(0)

    return paths, labels

def batch_forward(batch_paths,
                  conv1_filters, conv1_biases,
                  conv2_filters, conv2_biases,
                  conv3_filters, conv3_biases,
                  dense1_W, dense1_b,
                  dense2_W, dense2_b):
    outputs, caches = [], []

    for path in batch_paths:

        conv1_out, cache1 = conv_forward(path, num_filters=8, filter_size=3, stride=1, padding=1)
        cache1["filters"], cache1["biases"] = conv1_filters, conv1_biases
        pool1_out, cache_max1 = max_pooling(conv1_out, size=2, stride=2)

        conv2_out, cache2 = conv_forward_from_matrix(pool1_out, num_filters=16, filter_size=3, stride=1, padding=1)
        cache2["filters"], cache2["biases"] = conv2_filters, conv2_biases
        pool2_out, cache_max2 = max_pooling(conv2_out, size=2, stride=2)

        conv3_out, cache3 = conv_forward_from_matrix(pool2_out, num_filters=32, filter_size=3, stride=1, padding=1)
        cache3["filters"], cache3["biases"] = conv3_filters, conv3_biases
        pool3_out, cache_max3 = max_pooling(conv3_out, size=2, stride=2)


        flat, flatten_cache = flatten(pool3_out)

        z1 = flat @ dense1_W + dense1_b
        a1 = np.maximum(0, z1) 
        dense1_cache = {
            "input": flat, "weights": dense1_W, "biases": dense1_b,
            "z": z1, "activation": "ReLU"
        }

        z2 = a1 @ dense2_W + dense2_b
        a2 = 1 / (1 + np.exp(-z2)) 
        dense2_cache = {
            "input": a1, "weights": dense2_W, "biases": dense2_b,
            "z": z2, "activation": "sigmoid"
        }

        outputs.append(a2)
        caches.append([cache1, cache_max1, cache2, cache_max2, cache3, cache_max3,
                       flatten_cache, dense1_cache, None, dense2_cache])

    return outputs, caches


def batch_backward(y_true_batch, y_pred_batch, batch_caches):
    batch_size = len(y_true_batch)
    total_gradients = {}

    for idx in range(batch_size):
        y_true, y_pred = y_true_batch[idx], y_pred_batch[idx]
        caches = batch_caches[idx]

        dA = binary_crossentropy_gradient(y_true, y_pred)

        dA, dW, db = dense_backward(dA, caches[9])
        total_gradients.setdefault("dense2_dW", 0)
        total_gradients.setdefault("dense2_db", 0)
        total_gradients["dense2_dW"] += dW / batch_size
        total_gradients["dense2_db"] += db / batch_size

        dA = dropout_backward(dA, caches[8])

        dA, dW, db = dense_backward(dA, caches[7])
        total_gradients.setdefault("dense1_dW", 0)
        total_gradients.setdefault("dense1_db", 0)
        total_gradients["dense1_dW"] += dW / batch_size
        total_gradients["dense1_db"] += db / batch_size


        dA = flatten_backward(dA, caches[6])


        dA = max_pool_backward(dA, caches[5])

        dA, dW, db = conv_backward(dA, caches[4])
        total_gradients.setdefault("conv3_dW", 0)
        total_gradients.setdefault("conv3_db", 0)
        total_gradients["conv3_dW"] += dW / batch_size
        total_gradients["conv3_db"] += db / batch_size

        dA = max_pool_backward(dA, caches[3])


        dA, dW, db = conv_backward(dA, caches[2])
        total_gradients.setdefault("conv2_dW", 0)
        total_gradients.setdefault("conv2_db", 0)
        total_gradients["conv2_dW"] += dW / batch_size
        total_gradients["conv2_db"] += db / batch_size


        dA = max_pool_backward(dA, caches[1])


        dA, dW, db = conv_backward(dA, caches[0])
        total_gradients.setdefault("conv1_dW", 0)
        total_gradients.setdefault("conv1_db", 0)
        total_gradients["conv1_dW"] += dW / batch_size
        total_gradients["conv1_db"] += db / batch_size

    return total_gradients


learning_rate = 0.0001
epochs = 5
batch_size = 8
model_folder = r"D:\Project\MajorDatasets\MiniModelVariant\Model1"
log_file = r"D:\Project\training_log_2.txt"

paths, labels = load_dataset(model_folder)
num_samples = len(paths)
print(f"[{datetime.now()}] Loaded {num_samples} images")


conv1_filters, conv1_biases = initialize_conv_layer(8, 3, 3)
conv2_filters, conv2_biases = initialize_conv_layer(16, 3, 8)
conv3_filters, conv3_biases = initialize_conv_layer(32, 3, 16)

dummy_input, _ = conv_forward(paths[0], num_filters=8, filter_size=3, stride=1, padding=1)
dummy_input, _ = max_pooling(dummy_input, size=2, stride=2)
dummy_input, _ = conv_forward_from_matrix(dummy_input, num_filters=16, filter_size=3, stride=1, padding=1)
dummy_input, _ = max_pooling(dummy_input, size=2, stride=2)
dummy_input, _ = conv_forward_from_matrix(dummy_input, num_filters=32, filter_size=3, stride=1, padding=1)
dummy_input, _ = max_pooling(dummy_input, size=2, stride=2)
flat, _ = flatten(dummy_input)

dense1_out, dense1_cache = densen(flat, 128, activation="ReLU")
dense1_W, dense1_b = dense1_cache["weights"], dense1_cache["biases"]

dense2_out, dense2_cache = densen(dense1_out, 1, activation="sigmoid")
dense2_W, dense2_b = dense2_cache["weights"], dense2_cache["biases"]

print(f"[{datetime.now()}] Model weights initialized")

with open(log_file, 'w') as f:
    f.write("Epoch, Loss, Accuracy\n")


for epoch in tqdm(range(epochs), desc="Epochs"):
    indices = np.random.permutation(num_samples)
    paths = [paths[idx] for idx in indices]
    labels = [labels[idx] for idx in indices]

    total_loss, total_accuracy = 0, 0
    num_batches = num_samples // batch_size

    batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1} Batches", leave=False)
    for b in batch_pbar:
        batch_paths = paths[b*batch_size:(b+1)*batch_size]
        batch_labels = labels[b*batch_size:(b+1)*batch_size]

        batch_outputs, batch_caches = batch_forward(
            batch_paths,
            conv1_filters, conv1_biases,
            conv2_filters, conv2_biases,
            conv3_filters, conv3_biases,
            dense1_W, dense1_b,
            dense2_W, dense2_b
        )

        batch_preds_scalar = [float(out.flatten()[0]) for out in batch_outputs]
        loss = np.mean([binary_crossentropy_loss(batch_labels[idx], batch_preds_scalar[idx]) for idx in range(batch_size)])
        total_loss += loss

        preds = (np.array(batch_preds_scalar) > 0.5).astype(int)
        accuracy = np.mean(preds == np.array(batch_labels))
        total_accuracy += accuracy

        gradients = batch_backward(batch_labels, batch_outputs, batch_caches)

        conv1_filters -= learning_rate * gradients['conv1_dW']
        conv1_biases -= learning_rate * gradients['conv1_db']
        conv2_filters -= learning_rate * gradients['conv2_dW']
        conv2_biases -= learning_rate * gradients['conv2_db']
        conv3_filters -= learning_rate * gradients['conv3_dW']
        conv3_biases -= learning_rate * gradients['conv3_db']
        dense1_W -= learning_rate * gradients['dense1_dW']
        dense1_b -= learning_rate * gradients['dense1_db']
        dense2_W -= learning_rate * gradients['dense2_dW']
        dense2_b -= learning_rate * gradients['dense2_db']

        batch_pbar.set_postfix({'Batch Loss': f'{loss:.4f}', 'Batch Acc': f'{accuracy:.4f}'})

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    print(f"[{datetime.now()}] Epoch {epoch+1}: Avg Loss = {avg_loss}, Avg Accuracy = {avg_accuracy}")

    with open(log_file, 'a') as f:
        f.write(f"{epoch+1}, {avg_loss}, {avg_accuracy}\n")

print(f"[{datetime.now()}] Training complete! Log saved to training_log.txt")
