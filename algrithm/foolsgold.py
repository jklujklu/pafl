"""
Name: FoolsGold
Method: Cosine similarity
Cite: Fung, Clement, Chris JM Yoon, and Ivan Beschastnikh. "The Limitations of Federated Learning in Sybil Settings." 23rd International Symposium on Research in Attacks, Intrusions and Defenses ({RAID} 2020). 2020.
"""
import copy

import numpy as np
import sklearn.metrics.pairwise as smp

from algrithm.utils import multi_vectorization


def foolsgold(grads):
    n_clients = grads.shape[0]
    grads_bak = copy.deepcopy(grads)
    grads = multi_vectorization(grads)
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    # 扩展维度
    wv = wv[:, np.newaxis]
    wv /= wv.sum()
    # 聚合
    agg = np.sum(wv * grads_bak, axis=0)
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
    rs = foolsgold(predictions)
    print(rs)
