"""
Name: Krum
Method: Krum
Cite: Blanchard P, El Mhamdi E M, Guerraoui R, et al. Machine learning with adversaries: Byzantine tolerant gradient descent[J]. Advances in neural information processing systems, 2017, 30.
"""
import copy

import numpy as np

from algrithm.utils import cal_distance, compute_distance, multi_vectorization


def krum(grads, num_selected):
    num_models = len(grads)
    grads_bak = copy.deepcopy(grads)

    grads = multi_vectorization(grads)
    # 计算每个模型与其他模型的距离矩阵
    distances = cal_distance(grads)

    # 对于每个模型，选择与其距离最近的k-1个模型
    closest_indices = np.argsort(distances, axis=1)[:, 1:num_selected + 1]

    # 计算每个模型与选定的k-1个模型的平均距离
    avg_distances = np.zeros(num_models)
    for i in range(num_models):
        closest_preds = grads[closest_indices[i]]
        avg_distances[i] = np.mean(
            np.array([compute_distance(grads[i], closest_preds[j]) for j in range(num_selected)]))

    # 选择具有最小平均距离的模型作为最终可靠模型
    selected_model_index = np.argmin(avg_distances)

    return grads_bak[selected_model_index]


if __name__ == '__main__':
    model1_pred = np.random.rand(10)
    # 模型2的预测结果
    model2_pred = np.random.rand(10)
    # 模型3的预测结果
    model3_pred = np.random.rand(10)
    # 将所有模型的预测结果存储在一个列表中
    predictions = np.array([model2_pred, model1_pred, model3_pred])
    # 选择最可靠的模型
    selected_model = krum(predictions, num_selected=1)
    print(selected_model)
