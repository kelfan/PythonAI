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
