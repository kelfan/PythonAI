# 用来计算两点之间的距离的 
from scipy.spatial import distance 

def euc(a,b):
    return distance.euclidean(a,b)

# import random

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train 

    def predict(self, X_test): 
        predictions = []
        for row in X_test:
            # label = random.choice(self.y_train)
            # 找出跟训练集最接近的点,然后放到一类 label表示类型
            label = self.closest(row)
            # predictions 不断累加测试集的分类 [1,0,1...]
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        # 初始化,把第一个作为最开始的点
        best_dist = euc(row, self.X_train[0])
        best_index = 0 
        # 每一个都跟前面的最短的路径对比,得到更短的点,循环完后得到最短的点
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i]) # 计算两点之间的距离 
            if dist < best_dist:
                best_dist = dist 
                best_index = i 
        return self.y_train[best_index] #返回最短距离的点的类型 

# import a dataset 
from sklearn import datasets
iris = datasets.load_iris()

# 获取数据
X = iris.data
y = iris.target

# 拆分数据为测试集和训练集 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# 构建模型 分类器
from sklearn import tree 
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

# 通过模型进行预测 
predictions = my_classifier.predict(X_test)
print(predictions)

# 测试准度 
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test, predictions))
