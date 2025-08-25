import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from single_convolution_krishna import single_convolution_layer, single_convolution_layer_from_matrix, conv_backward
from max_pooling_krishna import max_pool, maxpool_backward
from forward_prop_layers import flatten, densen
from backward_prop_layers import dense_backward, flatten_backward, binary_crossentropy_gradient, binary_crossentropy_loss, dropout_backward

# New functions for multi-class
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def categorical_crossentropy_loss(y_true_onehot, y_pred):
    return -np.sum(y_true_onehot * np.log(y_pred + 1e-8))

def onehot(label, num_classes):
    oh = np.zeros(num_classes)
    oh[label] = 1
    return oh

def apply_dropout(X, rate):
    mask = np.random.binomial(1, 1 - rate, size=X.shape) / (1 - rate)
    out = X * mask
    cache = {'mask': mask}
    return out, cache

def batch_forward(batch_paths, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b, num_classes, dropout_rate=0.5):
    outputs = []
    batch_caches = []
    for path in batch_paths:
        # Layer 1: Conv + MaxPool
        conv1_out, cache1 = single_convolution_layer(path, filter_size=(3, 3), stride=1, num_filters=32, padding=1, filters=conv1_filters, biases=conv1_biases)
        pooled1, cache_pool1 = max_pool(conv1_out, size=(2, 2), stride=2)

        # Layer 2: Conv + MaxPool
        conv2_out, cache2 = single_convolution_layer_from_matrix(pooled1, filter_size=(3, 3), stride=1, num_filters=64, padding=1, filters=conv2_filters, biases=conv2_biases)
        pooled2, cache_pool2 = max_pool(conv2_out, size=(2, 2), stride=2)

        # Layer 3: Conv + MaxPool
        conv3_out, cache3 = single_convolution_layer_from_matrix(pooled2, filter_size=(3, 3), stride=1, num_filters=128, padding=1, filters=conv3_filters, biases=conv3_biases)
        pooled3, cache_pool3 = max_pool(conv3_out, size=(2, 2), stride=2)

        # Layer 4: Flatten + Dense hidden
        flattened, flatten_cache = flatten(pooled3)
        dense1, densen_cache1 = densen(flattened, 128, activation="ReLU", W=dense1_W, b=dense1_b)
        dense1_dropout, dropout_cache = apply_dropout(dense1, dropout_rate)

        # Layer 5: Output layer (logits)
        logits, densen_cache2 = densen(dense1_dropout, num_classes, activation="linear", W=dense2_W, b=dense2_b)
        output = softmax(logits)

        outputs.append(output)
        batch_caches.append([cache1, cache_pool1, cache2, cache_pool2, cache3, cache_pool3, flatten_cache, densen_cache1, dropout_cache, densen_cache2])
    return outputs, batch_caches


def batch_backward(y_true_batch, y_pred_batch, batch_caches, num_classes):
    batch_size = len(y_true_batch)
    total_gradients = {}
    for idx in range(batch_size):
        y_true = y_true_batch[idx]
        y_true_onehot = onehot(y_true, num_classes)
        y_pred = y_pred_batch[idx]
        dA = y_pred - y_true_onehot  # dL/dlogits (since activation='linear' for output)
        caches = batch_caches[idx]

        # Output layer
        dA, dW, db = dense_backward(dA, caches[9], 'linear')
        if idx == 0:
            total_gradients['dense2_dW'] = np.zeros_like(dW)
            total_gradients['dense2_db'] = np.zeros_like(db)
        total_gradients['dense2_dW'] += dW
        total_gradients['dense2_db'] += db

        # Dropout
        dA = dropout_backward(dA, caches[8])

        # Hidden dense
        dA, dW, db = dense_backward(dA, caches[7], 'relu')
        if idx == 0:
            total_gradients['dense1_dW'] = np.zeros_like(dW)
            total_gradients['dense1_db'] = np.zeros_like(db)
        total_gradients['dense1_dW'] += dW
        total_gradients['dense1_db'] += db

        # Flatten
        dA = flatten_backward(dA, caches[6])

        # MaxPool 3
        dA = maxpool_backward(dA, caches[5])

        # Conv 3
        dA, dW, db = conv_backward(dA, caches[4])
        if idx == 0:
            total_gradients['conv3_dW'] = np.zeros_like(dW)
            total_gradients['conv3_db'] = np.zeros_like(db)
        total_gradients['conv3_dW'] += dW
        total_gradients['conv3_db'] += db

        # MaxPool 2
        dA = maxpool_backward(dA, caches[3])

        # Conv 2
        dA, dW, db = conv_backward(dA, caches[2])
        if idx == 0:
            total_gradients['conv2_dW'] = np.zeros_like(dW)
            total_gradients['conv2_db'] = np.zeros_like(db)
        total_gradients['conv2_dW'] += dW
        total_gradients['conv2_db'] += db

        # MaxPool 1
        dA = maxpool_backward(dA, caches[1])

        # Conv 1
        dA, dW, db = conv_backward(dA, caches[0])
        if idx == 0:
            total_gradients['conv1_dW'] = np.zeros_like(dW)
            total_gradients['conv1_db'] = np.zeros_like(db)
        total_gradients['conv1_dW'] += dW
        total_gradients['conv1_db'] += db

    # Average gradients over batch
    for key in total_gradients:
        total_gradients[key] /= batch_size

    return total_gradients


def load_dataset(dataset_path):
    folders = ['CapitalLetters', 'SmallLetters', 'Numbers']
    class_names = sorted([str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            for subfolder in os.listdir(folder_path):
                sub_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(sub_path) and subfolder in class_to_idx:
                    idx = class_to_idx[subfolder]
                    for f in os.listdir(sub_path):
                        if f.lower().endswith('.ppm'):
                            paths.append(os.path.join(sub_path, f))
                            labels.append(idx)
    return paths, labels, class_names


def save_model(save_path, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b):
    np.savez(save_path,
             conv1_filters=conv1_filters, conv1_biases=conv1_biases,
             conv2_filters=conv2_filters, conv2_biases=conv2_biases,
             conv3_filters=conv3_filters, conv3_biases=conv3_biases,
             dense1_W=dense1_W, dense1_b=dense1_b,
             dense2_W=dense2_W, dense2_b=dense2_b)
    print(f"[{datetime.now()}] Model saved to {save_path}")


def load_model(load_path):
    data = np.load(load_path, allow_pickle=True)
    print(f"[{datetime.now()}] Model loaded from {load_path}")
    return (data['conv1_filters'], data['conv1_biases'],
            data['conv2_filters'], data['conv2_biases'],
            data['conv3_filters'], data['conv3_biases'],
            data['dense1_W'], data['dense1_b'],
            data['dense2_W'], data['dense2_b'])


def predict(image_path, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b, num_classes, class_names, dropout_rate=0.0):
    conv1_out, _ = single_convolution_layer(image_path, filter_size=(3, 3), stride=1, num_filters=32, padding=1, filters=conv1_filters, biases=conv1_biases)
    pooled1, _ = max_pool(conv1_out, size=(2, 2), stride=2)

    conv2_out, _ = single_convolution_layer_from_matrix(pooled1, filter_size=(3, 3), stride=1, num_filters=64, padding=1, filters=conv2_filters, biases=conv2_biases)
    pooled2, _ = max_pool(conv2_out, size=(2, 2), stride=2)

    conv3_out, _ = single_convolution_layer_from_matrix(pooled2, filter_size=(3, 3), stride=1, num_filters=128, padding=1, filters=conv3_filters, biases=conv3_biases)
    pooled3, _ = max_pool(conv3_out, size=(2, 2), stride=2)

    flattened, _ = flatten(pooled3)
    dense1, _ = densen(flattened, 128, activation="ReLU", W=dense1_W, b=dense1_b)
    dense1_dropout, _ = apply_dropout(dense1, dropout_rate)  # dropout_rate=0 for inference

    logits, _ = densen(dense1_dropout, num_classes, activation="linear", W=dense2_W, b=dense2_b)
    output = softmax(logits)
    pred_class_idx = np.argmax(output)
    pred_score = output[pred_class_idx]
    pred_class = class_names[pred_class_idx]
    return pred_score, pred_class


# ---------------- TRAINING ----------------
learning_rate = 0.01   # slightly higher, since model is smaller
epochs = 50
batch_size = 32   # Recommended batch size
num_classes = 62  # 10 numbers + 26 capital + 26 small
dataset_path = r"C:\Users\Krishna Gera\Desktop\Project Likhat\MajorDatasets\PPMFormatDataset"
log_file = r"C:\Users\Krishna Gera\Desktop\Project Likhat\Coding\training_log.txt"
save_path = r"C:\Users\Krishna Gera\Desktop\Project Likhat\Coding\trained_cnn_model_small.npz"

paths, labels, class_names = load_dataset(dataset_path)
num_samples = len(paths)
print(f"[{datetime.now()}] Loaded {num_samples} images")

# Weight initialization
init_path = paths[0]
conv1_out, cache1 = single_convolution_layer(init_path, filter_size=(3, 3), stride=1, num_filters=32, padding=1)
pooled1, cache_pool1 = max_pool(conv1_out, size=(2, 2), stride=2)
conv2_out, cache2 = single_convolution_layer_from_matrix(pooled1, filter_size=(3, 3), stride=1, num_filters=64, padding=1)
pooled2, cache_pool2 = max_pool(conv2_out, size=(2, 2), stride=2)
conv3_out, cache3 = single_convolution_layer_from_matrix(pooled2, filter_size=(3, 3), stride=1, num_filters=128, padding=1)
pooled3, cache_pool3 = max_pool(conv3_out, size=(2, 2), stride=2)
flattened, flatten_cache = flatten(pooled3)
dense1, densen_cache1 = densen(flattened, 128, activation="ReLU")
logits, densen_cache2 = densen(dense1, num_classes, activation="linear")

conv1_filters, conv1_biases = cache1['filters'], cache1['biases']
conv2_filters, conv2_biases = cache2['filters'], cache2['biases']
conv3_filters, conv3_biases = cache3['filters'], cache3['biases']
dense1_W, dense1_b = densen_cache1['weights'], densen_cache1['biases']
dense2_W, dense2_b = densen_cache2['weights'], densen_cache2['biases']

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

        batch_outputs, batch_caches = batch_forward(batch_paths, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b, num_classes)

        batch_preds = batch_outputs  # list of arrays
        loss = np.mean([categorical_crossentropy_loss(onehot(batch_labels[idx], num_classes), batch_preds[idx]) for idx in range(batch_size)])
        total_loss += loss

        preds = np.array([np.argmax(p) for p in batch_preds])
        accuracy = np.mean(preds == np.array(batch_labels))
        total_accuracy += accuracy

        gradients = batch_backward(batch_labels, batch_outputs, batch_caches, num_classes)

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

# Save trained model
save_model(save_path, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b)

print(f"[{datetime.now()}] Training complete! Log saved to training_log.txt")

# ---------------- TESTING EXAMPLE ----------------
(conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b) = load_model(save_path)

test_image = r"C:\Users\Krishna Gera\Desktop\Project Likhat\Coding\not_B_1.ppm"
prediction_score, predicted_class = predict(test_image, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b, num_classes, class_names)
print(f"Prediction score: {prediction_score:.4f}")
print("Class:", predicted_class)