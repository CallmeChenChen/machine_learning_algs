# -*- coding: utf-8 -*-
# @Date   : 2019/11/25
# @File   : NaiveBayes.py
# @Author : zhaochen

import numpy as np

'''
离散型变量的朴素贝叶斯
连续型变量的朴素贝叶斯
如果既有离散又有连续变量 朴素贝叶斯能实现吗？ 好像不能
'''

class NaiveBayesBase(object):

    def __init__(self, lamb=1):
        self.lamb = lamb

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class NaiveBayesDiscrete(NaiveBayesBase):

    def fit(self, X, y):

        return self.__discrete_features_estimate(X, y)

    def predict(self, X):
        n, m = X.shape
        prob = np.zeros((n, self.k))
        for i in range(n):
            for c in range(self.k):
                sample_condition_prob = 1
                for f in range(m):
                    X_f = X[i, f]
                    # 可以考虑取log防止数据下溢
                    condition_prob = self.condition_prob[c][f].get(X_f)
                    sample_condition_prob *= condition_prob if condition_prob is not None else self.condition_prob[c][f].get("unseen")
                prob[i, c] = self.prior_y[c] * sample_condition_prob
        label = np.argmax(prob, axis=1)
        return prob, label

    def __discrete_features_estimate(self, X, y):
        # 先验概率
        n, m = X.shape
        self.k = len(np.unique(y))
        count_y = np.bincount(y)
        self.prior_y = (count_y + self.lamb) / (n + self.k * self.lamb)
        # 条件概率 没想到怎么存能加速
        self.condition_prob = [[] for _ in range(self.k)]
        for c in range(self.k):
            X_c = X[y==c]
            for f in range(m):
                feature_condition_prob = {}
                X_f = X_c[:, f]
                nuniques_f = np.unique(X_f)
                for v in nuniques_f:
                    count_v = sum(X_f == v)
                    condition_prob_v = (count_v + self.lamb) / (count_y[c] + len(nuniques_f) * self.lamb)
                    feature_condition_prob[v] = condition_prob_v
                # feature_condition_prob[""] # 给未出现的feat value 赋予一个固定值
                feature_condition_prob["unseen"] = self.lamb / (count_y[c] + len(nuniques_f) * self.lamb)
                self.condition_prob[c].append(feature_condition_prob)
        return self


class NaiveBayesGaussian(NaiveBayesBase):

    def fit(self, X, y):
        return self.__continuous_features_estimate(X, y)

    def __continuous_features_estimate(self, X, y):
        n, m = X.shape
        self.k = len(np.unique(y))
        count_y = np.bincount(y)
        self.prior_y = count_y / n
        self.params_gaussian = [[] for _ in range(self.k)]
        for c in range(self.k):
            X_c = X[y==c]
            mu_ = np.mean(X_c, axis=0)
            sigma_ = np.std(X_c, axis=0)
            for f in range(m):
                self.params_gaussian[c].append({'mu':mu_[f],'sigma':sigma_[f]})
        return self

    def predict(self, X):
        n, m = X.shape
        prob = np.zeros((n, self.k))
        for i in range(n):
            for c in range(self.k):
                prior_prob = np.log(self.prior_y[c])
                gaussian_prob = 0
                for f in range(m):
                    gaussian_prob_f = self.gaussian(X[i,f], self.params_gaussian[c][f]['mu'], self.params_gaussian[c][f]['sigma'])
                    gaussian_prob += np.log(gaussian_prob_f)
                prob[i][c] = prior_prob + gaussian_prob
        label = np.argmax(prob, axis=1)
        return prob, label



    @staticmethod
    def gaussian(x, mu, sigma):
        '''from scipy.stats import norm
            norm(mu, sigma).pdf(x)
        '''
        coef = 1.0 / (np.sqrt(2 * np.pi * sigma))
        component = np.exp(-(x - mu)**2 / (2 * sigma ** 2))
        return coef * component



if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd
    FLAG = False
    if FLAG:
        data = pd.read_csv('../../data/car.data',
                           names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
        for col in data.columns:
            data[col], _ = pd.factorize(data[col])

        train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=2019)
        features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        clf = NaiveBayesDiscrete()
        clf.fit(train[features].values, y=train['label'].values)
        prob, label = clf.predict(test[features].values)
        print("accuracy: ",accuracy_score(y_true=test['label'], y_pred=label))
    else:
        data = pd.read_csv('../../data/PimaIndiansdiabetes.csv')
        features =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
        train, test = train_test_split(data, test_size=0.15, shuffle=True, random_state=2019)
        clf = NaiveBayesGaussian()
        clf.fit(train[features].values, train['Outcome'].values)
        prob, label = clf.predict(test[features].values)
        print("accuracy: ", accuracy_score(y_true=test['Outcome'], y_pred=label))

