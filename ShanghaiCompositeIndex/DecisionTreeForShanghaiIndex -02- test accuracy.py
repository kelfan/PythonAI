# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:01:48 2017

@author: Administrator
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Load a CSV file
def load_csv(filename):
    df = pd.read_csv(filename,
                  parse_dates = True,
                  index_col=0)
    return df

# split dataset 
def split_data(dataset,percentage):
    split =round(percentage*len(dataset))
    dataset_train = dataset[:split]
    dataset_test = dataset[split:]    
    return dataset_train,dataset_test 

# split train and test  
def split_train_test(dataset):
    train=dataset.values[:,:-2]
    test=dataset.values[:,-1]
    return train,test

# build the Decision Tree Model 
def build_decision_tree(dataset):
    # load and prepare data
    x,y=split_train_test(dataset)
    decision_tree=DecisionTreeClassifier()
    decision_tree.fit(x,y)
    return decision_tree

# load and prepare data
filename = 'shanghaiIndex.csv'
dataset = load_csv(filename)
data_train, data_test = split_data(dataset,0.5)
clf=build_decision_tree(data_test)
print(clf.score(data_test.values[:,:-2],data_test.values[:,-1]))

