"""
Name: RFA
Method: Geometric Median
Cite: Pillutla K, Kakade S M, Harchaoui Z. Robust aggregation for federated learning[J]. IEEE Transactions on Signal Processing, 2022, 70: 1142-1154.
"""
import copy

import numpy as np

from algrithm.utils import multi_vectorization


def geometric_median(models, epsilon=1e-5, max_iterations=100):
    num_models = len(models)
    num_params = models[0].shape[0]

    # 初始化几何中位数为第一个模型
    median = models[0]

    # 迭代更新几何中位数
    for _ in range(max_iterations):
        # 计算每个模型与当前几何中位数之间的差
        diffs = [model - median for model in models]

        # 计算欧氏距离的倒数
        inverse_distances = [1.0 / (np.linalg.norm(diff) + epsilon) for diff in multi_vectorization(diffs)]

        # 根据权重更新几何中位数
        weights_sum = sum(inverse_distances)
        new_median = 0
        for i in range(num_models):
            new_median += inverse_distances[i] * models[i]
        new_median /= weights_sum

        # 判断收敛条件
        tmp = multi_vectorization(new_median - median)
        if np.linalg.norm(np.concatenate(tmp)) < epsilon:
            break

        median = new_median

    return median


if __name__ == '__main__':
    # 假设有3个模型的参数，每个模型的参数形状为(10,)
    model1_pred = np.array([np.random.random((10, 10)), np.random.random((20, 30))])
    # 模型2的预测结果
    model2_pred = np.array([np.random.random((10, 10)), np.random.random((20, 30))])
    # 模型3的预测结果
    model3_pred = np.array([np.random.random((10, 10)), np.random.random((20, 30))])
    # model1_pred = np.random.rand(10)
    # model2_pred = np.random.rand(10)
    # model3_pred = np.random.rand(10)

    # 将模型参数存储在一个列表中
    models = [model1_pred, model2_pred, model3_pred]

    # 使用几何中位数进行模型参数的鲁棒聚合
    aggregated_params = geometric_median(models)

    # 打印聚合后的模型参数
    print(aggregated_params)
