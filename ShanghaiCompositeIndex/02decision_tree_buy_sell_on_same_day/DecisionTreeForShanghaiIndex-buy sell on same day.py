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
        self.train = data_train
        self.test = data_test
        train_x,train_y = self.split_feature_class(data_train)
        test_x, test_y = self.split_feature_class(data_test)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
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
    def split_feature_class(self, dataset):
        train=dataset.values[:,:-1]
        test=dataset.values[:,-1]
        return train,test
    
    # build the Decision Tree Model 
    def build_decision_tree(self, dataset):
        # load and prepare data
        x,y=self.split_feature_class(dataset)
        decision_tree=DecisionTreeClassifier()
        decision_tree.fit(x,y)
        return decision_tree
    
    # get score 
    def get_score(self, data):
        clf=self.build_decision_tree(data)
        x,y=self.split_feature_class(data)
        score = clf.score(x,y)
        return score

    # prediction
    def predict(self,data):
        result = self.model.predict(data)
        return result

# load and prepare data
filename = 'shanghaiIndex.csv'
dt = DecisionTreeFromCsv(filename)
predictions = dt.predict(dt.test_x)
print(dt.predict(dt.test_x))
print(dt.score)

print("-----------trade on the same next day------------")
Result = 0
for i in range(0,predictions.size-1):
    if("yes".__eq__(predictions.item(i))):
        tmp = dt.test_x[i+1][1] - dt.test_x[i+1][0]
        Result = Result + tmp
        print(Result)
