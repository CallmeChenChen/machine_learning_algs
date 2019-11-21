# -*- coding: utf-8 -*-
# @Date   : 2019/11/20
# @File   : KNN.py
# @Author : Cyril

import numpy as np

'''
1）计算测试数据与各个训练数据之间的距离；

2）按照距离的递增关系进行排序；

3）选取距离最小的K个点；

4）确定前K个点所在类别的出现频率；

5）返回前K个点中出现频率最高的类别作为测试数据的预测分类
'''

class KNN(object):

    def __init__(self, k, distance_mod='edu'):
        self.k = k
        self.cal_distance = self.euclidean_distance if distance_mod == 'edu' else self.manhattan_distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        '''
        测试数据 X
        :param X: X.shape n * m
        :return: array shape n * 1
        '''
        labels = []
        for i in range(len(X)):
            distance = self.cal_distance(self.X_train, X[i, :])
            argsort_distance = np.argsort(distance)[:self.k]
            k_labels = self.y_train[argsort_distance]
            mode_label = np.argmax(np.bincount(k_labels))
            labels.append(mode_label)
        return np.array(labels)

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum(np.square(np.array(x1) - np.array(x2)), axis=1))

    @staticmethod
    def manhattan_distance(x1, x2):
        return np.sum(np.abs(np.array((x1) - np.array(x2))), axis=1)


if __name__ == '__main__':

    from sklearn.datasets import load_iris
    iris = load_iris()
    X_train = iris['data']
    y_train = iris['target']

    knn = KNN(k=9)
    knn.fit(X_train, y_train)
    knn.predict(X_train[140:])

