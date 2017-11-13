# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:03:57 2017

@author: chaofanz
"""

file = "./iris.data.csv"
import pandas as pd 
df = pd.read_csv(file, header=None) # 没有标题或文件头
df.head(10) # 显示前10行

# 可视化显示 
import matplotlib.pyplot as plt 
import numpy as np 
y = df.loc[0:100, 4].values # 把0到100行的第4列的数据取出来
# print (y) 
y = np.where(y == 'Iris-setosa', -1, 1) # 把Iris-setosa等字符串转成1或-1
# print (y)
X = df.iloc[0:100, [0,2]].values # 把第0列和第2列的数据抽取出来
# 把x描绘出来 
plt.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('花瓣长度')
plt.ylabel('花茎长度')
plt.legend(loc='upper left')
plt.show()