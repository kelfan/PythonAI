# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:03:01 2017

@author: chaofanz
"""

import numpy as np

class Perceptron(object):
    '''
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    errors:用于记录神经元判断出错次数
    '''
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        pass
    
    def fit(self, X, y):
        '''
        输入训练数据， 培训神经元， X输入样本向量， y对应样本分类
        
        X:shape[n_samples, n_features]
        X:[[1, 2, 3], [4, 5, 6]]
        n_samples:2
        n_features:3
        
        y:[1, -1]
        '''
        """
        初始化权重向量为0
        加1是因为前面算法提到的w0, 也就是步调函数阈值，其实也就是偏置
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                """
                update = n * (y - y')
                """
                update = self.eta * (target - self.predict(xi))
                """
                xi 是一个向量
                """
                self.w_[1:] += update * xi
                self.w_[0] += update
                
                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
            pass
        
    def net_input(self, X):
        """
        z = W0*1 + W1*X1 + W2*X2 + ... Wn*Xn
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    pass
        
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    pass
    pass
import pandas as pd
df = pd.read_csv('iris.data.csv', header=None)
df.head(10)

import matplotlib.pyplot as plt
y = df.loc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1)

X = df.loc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('花瓣的长度')
plt.ylabel('花径的长度')
plt.legend(loc='upper left')

plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for inx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(inx),
                   marker=markers[inx], label=cl)
        
plot_decision_regions(X, y, ppn, resolution=0.02)

plt.xlabel('花径长度')
plt.ylabel('花瓣长度')
plt.legend(loc='upper left')
plt.show()