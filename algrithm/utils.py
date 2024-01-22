import copy

import numpy as np
from scipy.spatial.distance import euclidean


def multi_vectorization(grads):
    vectors = copy.deepcopy(grads)
    rs = []
    for i, v in enumerate(vectors):
        for j, layer in enumerate(v):
            v[j] = layer.reshape(-1)
        rs.append(np.hstack(v))
    return np.array(rs)


def compute_distance(a, b):
    return euclidean(a, b)


def cal_distance(grads):
    num_models = len(grads)
    # 计算每个模型与其他模型的距离矩阵
    distances = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(i + 1, num_models):
            distances[i, j] = compute_distance(grads[i], grads[j])
            distances[j, i] = distances[i, j]
    return distances
