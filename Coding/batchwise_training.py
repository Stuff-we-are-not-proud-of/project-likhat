import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from single_convolution_krishna import single_convolution_layer, single_convolution_layer_from_matrix, conv_backward
from max_pooling_krishna import max_pool, maxpool_backward
from forward_prop_layers import flatten, densen, dropout
from backward_prop_layers import dense_backward, dropout_backward, flatten_backward, binary_crossentropy_gradient, binary_crossentropy_loss

def batch_forward(batch_paths, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b):
    batch_size = len(batch_paths)
    outputs = []
    batch_caches = []
    for path in batch_paths:
        singly_convolved, cache1 = single_convolution_layer(path, filter_size=(3, 3), stride=1, num_filters=32, padding=1, filters=conv1_filters, biases=conv1_biases)
        singly_maxxed, cache_max1 = max_pool(singly_convolved, size=(2, 2), stride=2)
        doubly_convolved, cache2 = single_convolution_layer_from_matrix(singly_maxxed, filter_size=(3, 3), stride=1, num_filters=64, padding=1, filters=conv2_filters, biases=conv2_biases)
        doubly_maxxed, cache_max2 = max_pool(doubly_convolved, size=(2, 2), stride=2)
        triply_convolved, cache3 = single_convolution_layer_from_matrix(doubly_maxxed, filter_size=(3, 3), stride=1, num_filters=128, padding=1, filters=conv3_filters, biases=conv3_biases)
        triply_maxxed, cache_max3 = max_pool(triply_convolved, size=(2, 2), stride=2)
        flattened, flatten_cache = flatten(triply_maxxed)
        dense1, densen_cache = densen(flattened, 128, activation="ReLU", W=dense1_W, b=dense1_b)
        dropped, dropout_cache = dropout(dense1, dropout_rate=0.5, training=True)
        output, densen_cache_1 = densen(dropped, 1, activation="sigmoid", W=dense2_W, b=dense2_b)
        outputs.append(output)
        print(f"Sample pred: {output}")
        batch_caches.append([cache1, cache_max1, cache2, cache_max2, cache3, cache_max3, flatten_cache, densen_cache, dropout_cache, densen_cache_1])
    return outputs, batch_caches

def batch_backward(y_true_batch, y_pred_batch, batch_caches):
    batch_size = len(y_true_batch)
    total_gradients = {}
    for idx in range(batch_size):
        y_true = y_true_batch[idx]
        y_pred = y_pred_batch[idx]
        caches = batch_caches[idx]
        dA = binary_crossentropy_gradient(y_true, y_pred)
        
        dA, dW, db = dense_backward(dA, caches[9], 'sigmoid')
        if 'dense2_dW' not in total_gradients:
            total_gradients['dense2_dW'] = dW / batch_size
            total_gradients['dense2_db'] = db / batch_size
        else:
            total_gradients['dense2_dW'] += dW / batch_size
            total_gradients['dense2_db'] += db / batch_size
        
        dA = dropout_backward(dA, caches[8])
        
        dA, dW, db = dense_backward(dA, caches[7], 'relu')
        if 'dense1_dW' not in total_gradients:
            total_gradients['dense1_dW'] = dW / batch_size
            total_gradients['dense1_db'] = db / batch_size
        else:
            total_gradients['dense1_dW'] += dW / batch_size
            total_gradients['dense1_db'] += db / batch_size
        
        dA = flatten_backward(dA, caches[6])
        
        dA = maxpool_backward(dA, caches[5])
        
        dA, dW, db = conv_backward(dA, caches[4])
        if 'conv3_dW' not in total_gradients:
            total_gradients['conv3_dW'] = dW / batch_size
            total_gradients['conv3_db'] = db / batch_size
        else:
            total_gradients['conv3_dW'] += dW / batch_size
            total_gradients['conv3_db'] += db / batch_size
        
        dA = maxpool_backward(dA, caches[3])
        
        dA, dW, db = conv_backward(dA, caches[2])
        if 'conv2_dW' not in total_gradients:
            total_gradients['conv2_dW'] = dW / batch_size
            total_gradients['conv2_db'] = db / batch_size
        else:
            total_gradients['conv2_dW'] += dW / batch_size
            total_gradients['conv2_db'] += db / batch_size
        
        dA = maxpool_backward(dA, caches[1])
        
        dA, dW, db = conv_backward(dA, caches[0])
        if 'conv1_dW' not in total_gradients:
            total_gradients['conv1_dW'] = dW / batch_size
            total_gradients['conv1_db'] = db / batch_size
        else:
            total_gradients['conv1_dW'] += dW / batch_size
            total_gradients['conv1_db'] += db / batch_size
        
    return total_gradients

def load_dataset(model_folder):
    a_folder = os.path.join(model_folder, 'A')
    not_a_folder = os.path.join(model_folder, 'NotA')
    paths = []
    labels = []
    
    for f in os.listdir(a_folder):
        if f.lower().endswith('.ppm'):
            img_path = os.path.join(a_folder, f)
            paths.append(img_path)
            labels.append(1)
    
    for f in os.listdir(not_a_folder):
        if f.lower().endswith('.ppm'):
            img_path = os.path.join(not_a_folder, f)
            paths.append(img_path)
            labels.append(0)
    
    return paths, labels

learning_rate = 0.001
epochs = 5
batch_size = 8
model_folder = r"C:\Users\Krishna Gera\Desktop\Project Likhat\MajorDatasets\MiniModelVariant\Model1"
log_file = r"C:\Users\Krishna Gera\Desktop\Project Likhat\training_log.txt"

paths, labels = load_dataset(model_folder)
num_samples = len(paths)
print(f"[{datetime.now()}] Loaded {num_samples} images (162 A + 162 NotA)")

init_path = paths[0]
singly_convolved, cache1 = single_convolution_layer(init_path, filter_size=(3, 3), stride=1, num_filters=32, padding=1)
singly_maxxed, cache_max1 = max_pool(singly_convolved, size=(2, 2), stride=2)
doubly_convolved, cache2 = single_convolution_layer_from_matrix(singly_maxxed, filter_size=(3, 3), stride=1, num_filters=64, padding=1)
doubly_maxxed, cache_max2 = max_pool(doubly_convolved, size=(2, 2), stride=2)
triply_convolved, cache3 = single_convolution_layer_from_matrix(doubly_maxxed, filter_size=(3, 3), stride=1, num_filters=128, padding=1)
triply_maxxed, cache_max3 = max_pool(triply_convolved, size=(2, 2), stride=2)
flattened, flatten_cache = flatten(triply_maxxed)
dense1, densen_cache = densen(flattened, 128, activation="ReLU")
dropped, dropout_cache = dropout(dense1, dropout_rate=0.5, training=True)
output, densen_cache_1 = densen(dropped, 1, activation="sigmoid")

conv1_filters = cache1['filters']
conv1_biases = cache1['biases']
conv2_filters = cache2['filters']
conv2_biases = cache2['biases']
conv3_filters = cache3['filters']
conv3_biases = cache3['biases']
dense1_W = densen_cache['weights']
dense1_b = densen_cache['biases']
dense2_W = densen_cache_1['weights']
dense2_b = densen_cache_1['biases']

print(f"[{datetime.now()}] Model weights initialized")

with open(log_file, 'w') as f:
    f.write("Epoch, Loss, Accuracy\n")

for epoch in tqdm(range(epochs), desc="Epochs"):
    indices = np.random.permutation(num_samples)
    paths = [paths[idx] for idx in indices]
    labels = [labels[idx] for idx in indices]
    
    total_loss = 0
    total_accuracy = 0
    num_batches = num_samples // batch_size
    
    batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1} Batches", leave=False)
    for b in batch_pbar:
        batch_paths = paths[b*batch_size:(b+1)*batch_size]
        batch_labels = labels[b*batch_size:(b+1)*batch_size]
        
        print(f"[{datetime.now()}] Epoch {epoch+1}, Batch {b+1}/{num_batches}: Starting forward pass")
        batch_outputs, batch_caches = batch_forward(batch_paths, conv1_filters, conv1_biases, conv2_filters, conv2_biases, conv3_filters, conv3_biases, dense1_W, dense1_b, dense2_W, dense2_b)
        
        batch_preds_scalar = [out.flatten()[0] if isinstance(out, np.ndarray) else out for out in batch_outputs]
        loss = np.mean([binary_crossentropy_loss(batch_labels[idx], batch_preds_scalar[idx]) for idx in range(batch_size)])
        total_loss += loss
        
        preds = (np.array(batch_preds_scalar) > 0.5).astype(int)
        accuracy = np.mean(preds == np.array(batch_labels))
        total_accuracy += accuracy
        
        print(f"[{datetime.now()}] Epoch {epoch+1}, Batch {b+1}/{num_batches}: Loss = {loss}, Accuracy = {accuracy}. Starting backward pass")
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
        print(np.mean(gradients['dense2_dW']))
        
        batch_pbar.set_postfix({'Batch Loss': f'{loss:.4f}', 'Batch Acc': f'{accuracy:.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    print(f"[{datetime.now()}] Epoch {epoch+1}: Avg Loss = {avg_loss}, Avg Accuracy = {avg_accuracy}")
    
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1}, {avg_loss}, {avg_accuracy}\n")

print(f"[{datetime.now()}] Training complete! Log saved to training_log.txt")