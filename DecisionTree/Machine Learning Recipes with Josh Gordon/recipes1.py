# https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1

from sklearn import tree
# feature = [[140, "smooth"], [130,"smooth"],[170,"bumpy"]]
# labels = ["apple","apple","orange"]
feature = [[140, 1], [130,1],[170,0]]
labels = [0,0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature,labels)
print(clf.predict([[150,0]]))
# output [0]
