"""
Name: FoolsGold
Method: Cosine similarity
Cite: Fung, Clement, Chris JM Yoon, and Ivan Beschastnikh. "The Limitations of Federated Learning in Sybil Settings." 23rd International Symposium on Research in Attacks, Intrusions and Defenses ({RAID} 2020). 2020.
"""
import copy

import numpy as np
import sklearn.metrics.pairwise as smp

from algrithm.utils import multi_vectorization


def avg(grads):
    num = len(grads)
    agg = np.sum(grads, axis=0) / num
    return agg


if __name__ == '__main__':
    model1_pred = np.array([np.random.random((10, 10)), np.random.random((20, 30))])
    # 模型2的预测结果
    model2_pred = np.array([np.random.random((10, 10)), np.random.random((20, 30))])
    # 模型3的预测结果
    model3_pred = np.array([np.random.random((10, 10)), np.random.random((20, 30))])
    # 将所有模型的预测结果存储在一个列表中
    predictions = np.array([model2_pred, model1_pred, model3_pred])
    # 选择最可靠的模型
    rs = avg(predictions)
    print(rs)
