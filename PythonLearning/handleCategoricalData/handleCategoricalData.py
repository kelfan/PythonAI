# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:30:47 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import subprocess

from sklearn.tree import  export_graphviz


def visualize_tree(tree, feature_names):
    
    with open("dt.dot", 'w') as f:
        
        export_graphviz(tree, out_file=f, feature_names=feature_names,  filled=True, rounded=True, class_names=["Low", "High"] )

    command = ["C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe", "-Tpng", "C:\\Users\\Owner\\Desktop\\A\\Python_2016_A\\dt.dot", "-o", "dt.png"]
    
       
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")
    
data = pd.read_csv('adwords_data.csv', sep= ',' , header = 1)



X = data.values[:, [3,17,18,19,20,21,22]]
Y = data.values[:,8]

                           
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)                           
                           
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, 
                               max_depth=8, min_samples_leaf=4)
clf_gini.fit(X_train, y_train)   


visualize_tree(clf_gini, ["Words in Key Phrase", "AdSense",	"Mortgage",	"Money",	"loan", 	"lawyer", 	"attorney"])

