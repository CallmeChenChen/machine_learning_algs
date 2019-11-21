# -*- coding: utf-8 -*-
# @Date   : 2019/11/4
# @File   : KMeans.py
# @Author : Cyril

import numpy as np
import matplotlib.pyplot as plt


class KMeans(object):

    def __init__(self, k, distance_mod='euc', max_iter=None, min_sse=None):
        self.k = k
        self.max_iter = max_iter # 最大迭代次数
        self.min_error = min_sse if not None else 1000.0# 最小簇心变化距离
        if distance_mod == 'euc':
            self.cal_distance = self.euclidean_distance
        self.clusters = None

    def fit(self, data):
        '''终止条件：迭代次数，最小平方误差(Sum of Squared Error, 误差平方和)，簇中心变化率'''
        m, n = data.shape
        # 随机选择k个点作为初始中心
        centers = np.zeros((self.k, n))
        for i in range(self.k):
            idx = np.random.randint(0, len(data))
            centers[i, :] = data[idx, :]
        # 初始化每次迭代时，样本属于哪个簇 1列存属于哪类， 1列存误差
        clusters = np.zeros((m, 2))
        # 是否有最大迭代次数
        num_rounds = 0
        # 每次迭代的SSE
        SSEs = [0]
        while True:
            # 遍历所有样本, 找到样本所属的簇
            for i in range(m):
                # 对每个样本找到其所属最小距离的簇心
                for j in range(self.k):
                    distance = self.cal_distance(centers[j, :], data[i, :])
                    if j == 0:
                        min_distance = distance
                        min_part_idx = j
                    if distance < min_distance:
                        min_distance = distance
                        min_part_idx = j
                if clusters[i, min_part_idx] != min_part_idx:
                    clusters[i, 0], clusters[i, 1] = min_part_idx, min_distance**2  # 点到簇心平方和误差SSE
            # 更新 簇心, 用每个簇的新样本计算 簇心
            centers_change_error = [] # 簇心的变化
            centers_previous = centers # 记录之前的簇心
            for j in range(self.k):
                new_center_samples = data[clusters[:, 0] == j]
                centers[j, :] = np.mean(new_center_samples, axis=0)
                # 计算新的簇心 改变的距离
                centers_change_error.append(self.cal_distance(centers[j, :], centers_previous[j, :]))
            print(centers_change_error)
            SSE = sum(clusters[:, 1])
            num_rounds += 1
            print("SSE of num_rounds {}: ".format(num_rounds), SSE)
            if num_rounds >= self.max_iter or SSE <= self.min_error or SSEs[-1] == SSE:
                print("early stopping .")
                break
            SSEs.append(SSE)
        self.clusters = clusters
        return self

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum(np.square(np.array(x) - np.array(y))))
    @staticmethod
    def manhattan_distance(x, y):
        return np.sum(np.abs(np.array(x) - np.array(y)))
    @staticmethod
    def chebyshev_distance(x, y):
        return np.max(np.abs(np.array(x) - np.array(y)))
    @staticmethod
    def cos_distance(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))



if __name__ == '__main__':

    x = [2.273, 27.89, 30.519, 62.049, 29.263, 62.657, 75.735, 24.344, 17.667, 68.816, 69.076, 85.691]
    y = [68.367, 83.127, 61.07, 69.343, 68.748, 90.094, 62.761, 43.816, 86.765, 76.874, 57.829, 88.114]

    data = np.array((x, y)).T

    kmeans = KMeans(k=2, min_sse=30, max_iter=10)

    kmeans.fit(data)
    clusters = kmeans.clusters

    plt.scatter(x, y, c=clusters[:,0], marker='*')

