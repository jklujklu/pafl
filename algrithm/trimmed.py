"""
Name:
Method: Trimmed-mean
Cite: Yin D, Chen Y, Kannan R, et al. Byzantine-robust distributed learning: Towards optimal statistical rates[C]//International Conference on Machine Learning. PMLR, 2018: 5650-5659.
"""
import copy
from functools import reduce

import numpy as np

from algrithm.utils import cal_distance, multi_vectorization


def trimmed(grads, num_selected):
    grads_bak = copy.deepcopy(grads)

    grads = multi_vectorization(grads)
    # 计算每个模型与其他模型的距离矩阵
    distances = cal_distance(grads)

    distances = distances.sum(axis=0)
    med = np.median(distances)
    chosen = np.argsort(abs(distances - med))
    agg = reduce(lambda x,y: np.add(x,y), grads_bak[chosen[:num_selected]]) / num_selected
    return agg


if __name__ == '__main__':
    model1_pred = np.random.rand(10)
    # 模型2的预测结果
    model2_pred = np.random.rand(10)
    # 模型3的预测结果
    model3_pred = np.random.rand(10)
    # 将所有模型的预测结果存储在一个列表中
    predictions = np.array([model2_pred, model1_pred, model3_pred])
    # 选择最可靠的模型
    selected_model = trimmed(predictions, num_selected=2)
    print(selected_model)
