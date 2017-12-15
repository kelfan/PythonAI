# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:01:48 2017

@author: Administrator
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeFromCsv:
    
    def __init__(self, filename, split=0.5):
        self.filename = filename
        dataset = self.load_csv(filename)
        self.dataset = dataset
        data_train, data_test = self.split_data(dataset, split)
        clf = self.build_decision_tree(data_test)
        self.model = clf
        score = self.get_score(dataset)
        self.score = score

    # Load a CSV file
    def load_csv(self, filename):
        df = pd.read_csv(filename,
                      parse_dates = True,
                      index_col=0)
        return df
    
    # split dataset 
    def split_data(self,dataset,percentage):
        split =round(percentage*len(dataset))
        dataset_train = dataset[:split]
        dataset_test = dataset[split:]    
        return dataset_train,dataset_test 
    
    # split train and test  
    def split_train_test(self, dataset):
        train=dataset.values[:,:-2]
        test=dataset.values[:,-1]
        return train,test
    
    # build the Decision Tree Model 
    def build_decision_tree(self, dataset):
        # load and prepare data
        x,y=self.split_train_test(dataset)
        decision_tree=DecisionTreeClassifier()
        decision_tree.fit(x,y)
        return decision_tree
    
    # get score 
    def get_score(self, data):
        clf=self.build_decision_tree(data)
        x,y=self.split_train_test(data)
        score = clf.score(x,y)
        return score

# load and prepare data
filename = 'shanghaiIndex.csv'
# dataset = dt.load_csv(filename)
# data_train, data_test = dt.split_data(dataset,0.5)
# clf=build_decision_tree(data_test)
# print(clf.score(data_test.values[:,:-2],data_test.values[:,-1]))
dt = DecisionTreeFromCsv(filename)
print(dt.score)

