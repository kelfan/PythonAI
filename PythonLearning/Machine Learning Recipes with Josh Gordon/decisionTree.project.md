# terms 
Feature = 可以用来分类的特征 
Label = class,最后的分类种类

# simplest Tree classifier
```python
from sklearn import tree
# feature = [[140, "smooth"], [130,"smooth"],[170,"bumpy"]]
# labels = ["apple","apple","orange"]
feature = [[140, 1], [130,1],[170,0]]
labels = [0,0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature,labels)
print(clf.predict([[150,0]]))
# output [0]
```

# iris flower dataset = 用来训练对花进行分类的数据集 
https://en.wikipedia.org/wiki/Iris_flower_data_set

# scikit sample dataset 官方提供的数据集 
http://scikit-learn.org/stable/datasets/index.html

# graphviz 可视化数据工具 

# pyplot/ 生产柱状图 
```python
import numpy as np 
import matplotlib.pyplot as plt 

greyhounds = 500 
labs = 500 

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], 
        stacked=True, 
        color=['r','b'])
plt.show()
```

# SciPy/ distance = 计算两点之间的距离 
```python
from scipy.spatial import distance 
def euc(a,b):
    return distance.euclidean(a,b)
```

# TensorFlow for poets = 谷歌的开源关于花的分类器 
[TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)
[Train an Image Classifier with TensorFlow for Poets - Machine Learning Recipes #6](https://www.youtube.com/watch?v=cSKfRcEDGUs&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=6)

# TensorFlow/ Classifying handwritten digits using TensorFlow 识别手写数字 
[MNIST For ML Beginners &nbsp;|&nbsp; TensorFlow](https://www.tensorflow.org/get_started/mnist/beginners)
[Classifying Handwritten Digits with TF.Learn - Machine Learning Recipes #7](https://www.youtube.com/watch?v=Gj0iyo265bc&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=7)

# family Of decision Tree learning algorithm 
ID3 
C4.5 
C5.0 
CART = classification and regression trees 

# CART = classification and regression trees 
- 优先把单一的结果分离出来 -> 使用 impurity [impurity=0表示数据没有mix]
- information gain 来确定每次分离的问题的比重 
-  每个问题答案只有true/false,然后每行遍历一次
[Let’s Write a Decision Tree Classifier from Scratch - Machine Learning Recipes #8](https://www.youtube.com/watch?v=LDRbO9a6XPU)

# gini 是 impurity = 表示结果的混淆程度,含有别的可能 
# impurity = 用来衡量结果的纯度,有没有混有别的可能, 为0表示没有
[YouTube](https://www.youtube.com/watch?v=LDRbO9a6XPU)

# facets = 数据可视化工具 
[301 Moved Permanently](https://pair-code.github.io/facets)

# bucket = 把数据分成几段 
[YouTube](https://www.youtube.com/watch?v=d12ra3b_M-0)

# Feature crossing = 创建一个两两相交的表格来代表新的Feature
[YouTube](https://www.youtube.com/watch?v=d12ra3b_M-0)

# hashing  = 创建哈希值
# embedding = 把字变成向量vector来计算相似度 

# source 
[Machine Learning Recipes with Josh Gordon](https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)