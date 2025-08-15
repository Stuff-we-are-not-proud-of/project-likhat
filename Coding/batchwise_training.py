import numpy as np
import os
from single_convolution_krishna import single_convolution_layer, single_convolution_layer_from_matrix, conv_backward
from max_pooling_krishna import max_pool, maxpool_backward
from forward_prop_layers import flatten, densen, dropout
from backward_prop_layers import dense_backward, dropout_backward, flatten_backward, binary_crossentropy_gradient, binary_crossentropy_loss

def batch_forward(batch_paths):
    batch_size = len(batch_paths)
    outputs = []
    batch_caches = []
    for path in batch_paths:
        singly_convolved, cache1 = single_convolution_layer(path, filter_size=(3, 3), stride=1, num_filters=32, padding=1)
        singly_maxxed, cache_max1 = max_pool(singly_convolved, size=(2, 2), stride=2)
        doubly_convolved, cache2 = single_convolution_layer_from_matrix(singly_maxxed, filter_size=(3, 3), stride=1, num_filters=64, padding=1)
        doubly_maxxed, cache_max2 = max_pool(doubly_convolved, size=(2, 2), stride=2)
        triply_convolved, cache3 = single_convolution_layer_from_matrix(doubly_maxxed, filter_size=(3, 3), stride=1, num_filters=128, padding=1)
        triply_maxxed, cache_max3 = max_pool(triply_convolved, size=(2, 2), stride=2)
        flattened, flatten_cache = flatten(triply_maxxed)
        dense1, densen_cache = densen(flattened, 128, activation="ReLU")
        dropped, dropout_cache = dropout(dense1, dropout_rate=0.5, training=True)
        output, densen_cache_1 = densen(dropped, 1, activation="sigmoid")
        outputs.append(output)
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
epochs = 50
batch_size = 32
model_folder = r"C:\Users\Krishna Gera\Desktop\Project Likhat\MajorDatasets\MiniModelVariant\Model1"
log_file = r"C:\Users\Krishna Gera\Desktop\Project Likhat\training_log.txt"

paths, labels = load_dataset(model_folder)
num_samples = len(paths)
print(f"Loaded {num_samples} images (162 A + 162 NotA)")

with open(log_file, 'w') as f:
    f.write("Epoch, Loss, Accuracy\n")

for epoch in range(epochs):
    indices = np.random.permutation(num_samples)
    paths = [paths[idx] for idx in indices]
    labels = [labels[idx] for idx in indices]
    
    total_loss = 0
    total_accuracy = 0
    num_batches = num_samples // batch_size
    
    for b in range(num_batches):
        batch_paths = paths[b*batch_size:(b+1)*batch_size]
        batch_labels = labels[b*batch_size:(b+1)*batch_size]
        
        batch_outputs, batch_caches = batch_forward(batch_paths)
        
        loss = np.mean([binary_crossentropy_loss(batch_labels[idx], batch_outputs[idx]) for idx in range(len(batch_outputs))])
        total_loss += loss
        
        preds = np.array(batch_outputs) > 0.5
        accuracy = np.mean(preds == batch_labels)
        total_accuracy += accuracy
        
        gradients = batch_backward(batch_labels, batch_outputs, batch_caches)
        
        conv1_filters -= learning_rate * gradients['conv1_dW']
        conv1_biases -= learning_rate * gradients['conv1_db']
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    print(f"Epoch {epoch+1}: Loss = {avg_loss}, Accuracy = {avg_accuracy}")
    
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1}, {avg_loss}, {avg_accuracy}\n")

print("Training complete! Log saved to training_log.txt")