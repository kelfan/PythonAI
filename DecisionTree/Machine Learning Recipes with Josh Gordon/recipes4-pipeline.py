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
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

# 通过模型进行预测 
predictions = my_classifier.predict(X_test)
print(predictions)

# 测试准度 
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test, predictions))

# 通过KNeighbors构建模型 
from sklearn.neighbors import KNeighborsClassifier
my_classifier2 = KNeighborsClassifier()
my_classifier2.fit(X_train, y_train)

# 通过模型进行预测 
predictions2 = my_classifier2.predict(X_test)
print(predictions2)

# 测试准度 
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test, predictions2))