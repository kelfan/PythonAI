# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:53:31 2017

@author: chaofanz

运行环境 Anaconda Navigator Python 3
代码 https://github.com/FFGF/ML

只有一个类 Perceptron 
    函数 net_input(self, x) 算向量点积
    函数 predict(self, X) 激活函数算出1 或 -1
    函数 range(self.n_iter) 训练过程
    
"""
# 引入相关类库
import numpy as np 
# 定义感知器类 
class Perceptron(object):
    # 定义初始化函数 
    """
    eta: 学习率
    n_iter: 权重向量的训练次数
    w_: 神经分叉权重向量
    errors_: 用于记录神经元判断出错次数， 神经元预测错误数，根据这个数来判断神经元的预测效果
    """
    def __init__(self, eta = 0.01, n_iter=10):
        # 简单初始化 
        self.eta = eta 
        self.n_iter = n_iter
        pass
    # 函数：根据输入的样本进行神经元的培训
    def fit(self, X, y):
        """
        输入训练数据，培训神经元，X输入样本向量，y对应样本分类
        
        X：shape[n_samples, n_features]
        X:[[1,2,3], [4,5,6]]
        n_samples： 2 因为有两个向量
        n_features: 3 因为每个向量有3个电信号
        
        y:[1, -1] 表示分类，前面[1,2,3]就是分类1， 后面[4,5,6]就是分类 -1
        """
        
        # 初始化权重向量为 0 
        """
        W_ 就是权重 
        np.zero： numpy类库函数全部初始化为 0 
        X.shape[1] 就是 n_samples 就是向量个数 
        (1+X.shape[1]) 加一是因为前面算法提到的w0，也就是步调函数阈值
        """
        self.w_ =np.zeros(1+X.shape[1]);
        self.errors_ = [] # 错误向量作为空的数组
        
        # 进入训练过程
        for _ in range(self.n_iter): # 训练 n_iter 次，如果训练这么多次还不行的话，就终止
            errors = 0 
            """
            X:[1,2,3],[4,5,6]
            y:[1, -1]
            zip(X,y) = [1,2,3, 1], [4,5,6, -1]
            """
            for xi, target in zip(X,y):
                """
                update = η * （y - y') 
                η \eta 是学习率
                """
                update = self.eta * (target - self.predict(xi))
                """
                xi 是向量 
                update 是常量
                update * xi 等价 (ΔW(1) = X[1]*udpate, Δw(2) = X[2]*update, ΔW(3) = X[3]*update)
                [1:] 表示忽略第0个元素，从第1个元素开始
                """
                self.w_[1:] += update * xi
                self.w_[0] += update #阈值更新 w 的第0个元素是阈值 
                
                """
                如果出现错误，统计错误的次数
                错误的次数越来越少，模型的效果就越来越好，数据预测就越来越准确
                """
                errors += int(update != 0.0 )
                # 错误的统计次数放到错误列表中
                self.errors_.append(errors)
                pass
            pass
        # 数据的弱化，把输入数据和权重向量进行点积 
        """
        z= w0*1 + w1*x1 + ... Wn*Xn 
        """
        def net_input(self, X) :
            return np.dot(X, self.w_[1:]) + self.w_[0]
            pass
        
        # 分类预测，大于0就是1，小于0就是-1
        def predict(self, X):
            return np.where(self.net_input(X) >= 0.0 , 1, -1)
            pass
        pass

# 可视化显示数据

file = "./iris.data.csv"
import pandas as pd 
df = pd.read_csv(file, header=None) # 没有标题或文件头
df.head(10) # 显示前10行

# 可视化显示 
import matplotlib.pyplot as plt 
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
# plt.show()

# 输入对象
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数')
plt.show()

# 对数据进行预测和分类
from matplotlib.colors import ListedColormap  # 引入相关类库
def plot_decision_regions(X, y, classifier, resolution=0.02): # 定义函数
    markers=('s', 'x', 'o', 'v') # 数据显示的相关信息
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # 数据显示的颜色
    cmap = ListedColormap(colors[:len(np.unique(y))]) # 根据向量y的不同结果显示不同的颜色，向量y的结果是 1或-1，种类为2种，len为2，
    
    """
    X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组的所有第一维，第二维中取第0个数据，
    直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
    """                                       
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() # 获得花瓣长度的最大值和最小值
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() # 获得花径长度的最大值和最小值
    
    print(x1_min, x1_max)
    print(x2_min, x2_max)
   
    # xx1 是根据 x1_min, x1_max, resolution 的向量扩展为一个矩阵，xx2 同理
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    print(np.arange(x1_min, x1_max, resolution).shape)
    print(np.arange(x2_min, x2_max, resolution))
    print(xx1.shape)
    print(xx1)
    
#    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#    print(xx1.ravel())
#    print(xx2.ravel())
#    print(Z)

    # z存储模型分类后的结果 
    # ravel()把扩展后的多维向量还原为单维向量 
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(Z)
    
    # 绘制相关信息 
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx),
                   marker=markers[idx], label=cl)

plot_decision_regions(X, y, ppn, resolution=0.02)
plt.xlabel('花径长度')
plt.ylabel('花瓣长度')
plt.legend(loc='upper left')
plt.show()
"""
output:
# print(x1_min, x1_max)
# print(x2_min, x2_max)
3.3 7.0
0.0 5.1
# print(np.arange(x1_min, x1_max, resolution).shape)
# print(np.arange(x2_min, x2_max, resolution))
# 构造单维的有185个元素的向量
# 总共产生了185个元素，每个元素刚好相差一个resolution=0.02
(185,)
[ 3.3   3.32  3.34  3.36  3.38  3.4   3.42  3.44  3.46  3.48  3.5   3.52
  3.54  3.56  3.58  3.6   3.62  3.64  3.66  3.68  3.7   3.72  3.74  3.76
  3.78  3.8   3.82  3.84  3.86  3.88  3.9   3.92  3.94  3.96  3.98  4.
  4.02  4.04  4.06  4.08  4.1   4.12  4.14  4.16  4.18  4.2   4.22  4.24
  4.26  4.28  4.3   4.32  4.34  4.36  4.38  4.4   4.42  4.44  4.46  4.48
  4.5   4.52  4.54  4.56  4.58  4.6   4.62  4.64  4.66  4.68  4.7   4.72
  4.74  4.76  4.78  4.8   4.82  4.84  4.86  4.88  4.9   4.92  4.94  4.96
  4.98  5.    5.02  5.04  5.06  5.08  5.1   5.12  5.14  5.16  5.18  5.2
  5.22  5.24  5.26  5.28  5.3   5.32  5.34  5.36  5.38  5.4   5.42  5.44
  5.46  5.48  5.5   5.52  5.54  5.56  5.58  5.6   5.62  5.64  5.66  5.68
  5.7   5.72  5.74  5.76  5.78  5.8   5.82  5.84  5.86  5.88  5.9   5.92
  5.94  5.96  5.98  6.    6.02  6.04  6.06  6.08  6.1   6.12  6.14  6.16
  6.18  6.2   6.22  6.24  6.26  6.28  6.3   6.32  6.34  6.36  6.38  6.4
  6.42  6.44  6.46  6.48  6.5   6.52  6.54  6.56  6.58  6.6   6.62  6.64
  6.66  6.68  6.7   6.72  6.74  6.76  6.78  6.8   6.82  6.84  6.86  6.88
  6.9   6.92  6.94  6.96  6.98]
# print(xx1.shape)
# print(xx1)
(255, 185)  # 255行 185列
[[ 3.3   3.32  3.34 ...,  6.94  6.96  6.98] # 相当于前面的单维185个元素
 [ 3.3   3.32  3.34 ...,  6.94  6.96  6.98] # 下面全是重复
 [ 3.3   3.32  3.34 ...,  6.94  6.96  6.98]
 ..., 
 [ 3.3   3.32  3.34 ...,  6.94  6.96  6.98]
 [ 3.3   3.32  3.34 ...,  6.94  6.96  6.98]
 [ 3.3   3.32  3.34 ...,  6.94  6.96  6.98]]
# print(np.arange(x2_min, x2_max, resolution).shape)
# print(np.arange(x2_min, x2_max, resolution))
# 255个向量，每个刚好相差0.02 =》 因为从最小值到最大值按0.02隔开得到的数等于255个
(255,)
[ 0.    0.02  0.04  0.06  0.08  0.1   0.12  0.14  0.16  0.18  0.2   0.22
  0.24  0.26  0.28  0.3   0.32  0.34  0.36  0.38  0.4   0.42  0.44  0.46
  0.48  0.5   0.52  0.54  0.56  0.58  0.6   0.62  0.64  0.66  0.68  0.7
  0.72  0.74  0.76  0.78  0.8   0.82  0.84  0.86  0.88  0.9   0.92  0.94
  0.96  0.98  1.    1.02  1.04  1.06  1.08  1.1   1.12  1.14  1.16  1.18
  1.2   1.22  1.24  1.26  1.28  1.3   1.32  1.34  1.36  1.38  1.4   1.42
  1.44  1.46  1.48  1.5   1.52  1.54  1.56  1.58  1.6   1.62  1.64  1.66
  1.68  1.7   1.72  1.74  1.76  1.78  1.8   1.82  1.84  1.86  1.88  1.9
  1.92  1.94  1.96  1.98  2.    2.02  2.04  2.06  2.08  2.1   2.12  2.14
  2.16  2.18  2.2   2.22  2.24  2.26  2.28  2.3   2.32  2.34  2.36  2.38
  2.4   2.42  2.44  2.46  2.48  2.5   2.52  2.54  2.56  2.58  2.6   2.62
  2.64  2.66  2.68  2.7   2.72  2.74  2.76  2.78  2.8   2.82  2.84  2.86
  2.88  2.9   2.92  2.94  2.96  2.98  3.    3.02  3.04  3.06  3.08  3.1
  3.12  3.14  3.16  3.18  3.2   3.22  3.24  3.26  3.28  3.3   3.32  3.34
  3.36  3.38  3.4   3.42  3.44  3.46  3.48  3.5   3.52  3.54  3.56  3.58
  3.6   3.62  3.64  3.66  3.68  3.7   3.72  3.74  3.76  3.78  3.8   3.82
  3.84  3.86  3.88  3.9   3.92  3.94  3.96  3.98  4.    4.02  4.04  4.06
  4.08  4.1   4.12  4.14  4.16  4.18  4.2   4.22  4.24  4.26  4.28  4.3
  4.32  4.34  4.36  4.38  4.4   4.42  4.44  4.46  4.48  4.5   4.52  4.54
  4.56  4.58  4.6   4.62  4.64  4.66  4.68  4.7   4.72  4.74  4.76  4.78
  4.8   4.82  4.84  4.86  4.88  4.9   4.92  4.94  4.96  4.98  5.    5.02
  5.04  5.06  5.08]
# print(xx2.shape)
# print(xx2)
(255, 185) # 255行 185列
[[ 0.    0.    0.   ...,  0.    0.    0.  ] # 重复第一个元素185次
 [ 0.02  0.02  0.02 ...,  0.02  0.02  0.02] # 重复第二个元素185次
 [ 0.04  0.04  0.04 ...,  0.04  0.04  0.04]
 ..., 
 [ 5.04  5.04  5.04 ...,  5.04  5.04  5.04]
 [ 5.06  5.06  5.06 ...,  5.06  5.06  5.06]
 [ 5.08  5.08  5.08 ...,  5.08  5.08  5.08]] # 重复255行

# print(xx1.ravel())
# print(xx2.ravel())
# print(Z)
# 根据向量 xx1和xx2 得到预测分类 1和-1 如 3.3 和 0 得到的是 -1； 
[ 3.3   3.32  3.34 ...,  6.94  6.96  6.98]
[ 0.    0.    0.   ...,  5.08  5.08  5.08]
[-1 -1 -1 ...,  1  1  1]

"""