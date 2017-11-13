import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

df = pd.read_csv('iris.data.csv', header=None)
y = df.loc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1)

X = df.loc[0:100, [0, 2]].values

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

# 主要的内容在这个类里面
class AdalineGD(object):
    """
    eta:float
    学习效率，处于0和1
    
    n_iter:int
    对训练数据进行学习改进次数
    
    w_:一维向量
    存储权重数值
    
    error_:
    存储每次迭代改进时，网络对数据进行错误判断的次数
    """
    
    # 初始化函数
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    
    # 训练函数
    def fit(self, X, y):
        """
        X:二维数组[n_samples, n_features]
        n_samples 表示X中含有训练数据条目数
        n_features 表示每天数据，含有4个数据的一维向量，用于表示一条训练数目
        
        y:一维向量
        用于存储每一训练条目对应的正确分类
        """
        # 权重全部初始化为零
        self.w_ = np.zeros(1 + X.shape[1])
        # 记录运算结果的成本（正确结果和计算结果之差）来显示效率
        self.cost_ = []
        
        # 对权重w进行改进
        # 这个方法会让结果不断向cost成本最小化前进
        for i in range(self.n_iter):
            # 把当前数据和权重做乘积
            output = self.net_input(X)
            # output = w0 + w1*x1 + ... + wn*xn
            
            # 正确结果-算出的结果
            errors = (y - output)
            
            # 更新权重 
            self.w_[1:] += self.eta * X.T.dot(errors)
            # += 增量值
            # X.T.dot(errors) 是求偏导
            # X.T.dot() 是做转置再做点乘
            
            # 改进最后的w0
            self.w_[0] += self.eta * errors.sum()
            
            #每次改进后的成本, 数值越来越小说明改进越来越有效
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
            # 加入到成本列表
        return self
    
    # 乘积求和
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    # 激活函数
    def activation(self, X):
        return self.net_input(X)
    
    # 预测函数
    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1 )
        # 如果结果大于0就是分类1，否则分类-1
    
# 初始化 学习率为0.0001 训练次数50 学习率越小越好，训练次数越高越好，不过效率就越差
ada = AdalineGD(eta=0.0001, n_iter=50)

# X是训练数据，y是对应的结果
ada.fit(X, y)

# 构建一组预测数据，数据输入到模型，模型进行分类
plot_decision_regions(X, y, classifier=ada, resolution=0.02)
# 分类器使用训练的数据 ada
# resolution 

# 描绘结果
plt.title('Adaline-Gradient descent')
plt.xlabel('花径长度')
plt.ylabel('花瓣的长度')
plt.legend(loc='upper left')
plt.show()

# 改进的图显示
plt.plot(range(1, 51), ada.cost_, marker='o')
plt.plot(range(1, len(ada.cost_), ada.cost_, marker='o'))
plt.plot('Epochs')
plt.ylabel('sum-squard-error')
plt.show()