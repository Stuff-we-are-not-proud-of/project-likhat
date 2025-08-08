import numpy as np
def max_pooling(feature_map, size=2, stride=2):
    H_in, W_in, C = feature_map.shape
    H_out = (H_in - size) // stride + 1
    W_out = (W_in - size) // stride + 1
    pooled = np.zeros((H_out, W_out, C))

    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = min(h_start + size, H_in)
                w_start = j * stride
                w_end = min(w_start + size, W_in)

                region = feature_map[h_start:h_end, w_start:w_end, c]
                if region.size > 0:
                    pooled[i, j, c] = np.max(region)
    cache = {
        "input": feature_map,
        "size": size,
        "stride": stride,
        "pooled_shape": pooled.shape
    }
    return pooled, cache
