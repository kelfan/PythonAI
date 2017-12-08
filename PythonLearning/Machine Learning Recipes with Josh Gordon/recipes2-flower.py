"""
1. import dataset 
2. train a classifier 
3. predict Label for new flower
4. visualize the Tree
"""
# 1. import dataset 
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
# 打印第一个数据
print(iris.data[0])
# 打印第一个花的分类归属
print(iris.target[0])

# 打印整个数据集 
for i in range(len(iris.target)):
    print("Example %d: Label %s, Feature %s" % (i, iris.target[i], iris.data[i]))

# 2. train a classifier 
import numpy as np 
test_idx = [0,50,100]

# training data 
train_target = np.delete(iris.target, test_idx) #删除某些行
train_data = np.delete(iris.data, test_idx, axis=0) #删除某些行

# testing data 
test_target = iris.target[test_idx] #只取某些行
test_data = iris.data[test_idx] #只取某些行 

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# 3. predict Label for new flower
# prediction should be the same as target 
print(test_target)
print(clf.predict(test_data))

# 4. visualize the Tree
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("iris.pdf")