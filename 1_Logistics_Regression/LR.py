# -*- coding: utf-8 -*-
# @Date   : 2019/09/18
# @File   : KNN.py
# @Author : Cyril

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class LR(object):
    '''实现逻辑回归'''

    def __init__(self, learning_rate=0.1, max_iter=1000, batch_size=None):

        self.lr = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.batch_size = batch_size

    def train(self, X, y):
        '''
        X:train data y:train label
        参数更新
        返回 训练好的权重
        '''
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values

        W, b = self.weight_initialize(X)

        cost_list = []
        for i in range(self.max_iter):
            if not self.batch_size:
                X_ = X
                y_ = y
            else:
                idx = np.random.randint(0, len(X), self.batch_size)
                X_ = X[idx]
                y_ = y[idx]

            y_head, cost, dW, db = self.weight_gradient(X_, y_, W, b)
            W = W - self.lr * dW
            b = b - self.lr * db

            if i % 100:
                cost_list.append(cost)
                print('Train loss {0} for {1} iter'.format(cost, i))
            params = {'W':W, 'b':b}
            grads = {'dW':db, 'db':db}
        print('W and b:', params)
        print('dW and db:', grads)
        self.weights = params
        return self

    def predict(self, X):
        '''
        X: test data
        用训练好的模型进行预测
        '''
        W = self.weights['W']
        b = self.weights['b']
        y_pred = sigmoid(np.dot(X, W) + b)
        y_pred = np.where(y_pred > 0.5, 1, 0).squeeze()
        return y_pred

    @staticmethod
    def weight_initialize(x):
        W = np.zeros((x.shape[1], 1))
        b = 0
        return W, b
    @staticmethod
    def weight_gradient(X, y, W, b):
        if y.shape != (len(y), 1):
            y = y.reshape(-1,1)
        num = len(X)
        y_head = sigmoid(np.dot(X, W) + b)
        # print('y_head shape', y_head.shape)
        # tmp = y_head - y
        # 定义损失函数
        cost = - 1.0 / num * np.sum(y * np.log(y_head) + (1 - y) * np.log(1 - y_head))
        # 对参数求导
        dW = np.dot(X.T, (y_head - y)) / num
        db = np.sum(y_head - y) / num
        # 返回损失值
        cost = np.squeeze(cost)
        return y_head, cost, dW, db


if __name__ == '__main__':

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    y = np.where(y <=1, 0, 1)

    clf = LR(learning_rate=0.01, max_iter=100, batch_size=50)
    clf.train(X, y)
    y_pred = clf.predict(X)
    print('auc', roc_auc_score(y, y_pred))

