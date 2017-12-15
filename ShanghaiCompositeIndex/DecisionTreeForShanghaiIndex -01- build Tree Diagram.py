# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:01:48 2017

@author: Administrator
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# Load a CSV file
def load_csv(filename):
    df  = pd.read_csv(filename,
                  parse_dates = True,
                  index_col=0)
    return df

def build_decision_tree(dataset):
    # load and prepare data
    x=dataset.values[:,:-1]
    y=dataset.values[:,-1]
    decision_tree=DecisionTreeClassifier()
    decision_tree.fit(x,y)
    return decision_tree


# load and prepare data
filename = 'shanghaiIndex.csv'
dataset = load_csv(filename)
clf=build_decision_tree(dataset)
titles = list(dataset)[:-2]

# export the decsion tree into a graph file 
dot_data = StringIO()
class_names = ['no','yes']

export_graphviz(clf, 
                out_file=dot_data,
                filled=True, 
                rounded=True,
                special_characters=True,
                feature_names=titles,
                class_names=class_names,
                node_ids=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png('E:\workspace\PythonAI\ShanghaiCompositeIndex\dtTest.png')