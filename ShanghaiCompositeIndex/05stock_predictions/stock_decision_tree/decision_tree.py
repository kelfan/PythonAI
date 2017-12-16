# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:01:48 2017

@author: Administrator
"""
import pandas as pd
from datetime import date
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import pickle
import file_handler

# data_x, data_y should dataframe format
class decision_tree_model:
    
    def __init__(self, data_x, data_y,stock_code, split=0.5, times=50,folder='./models/'):
        data_x = data_x[:-1]
        if(len(data_x.index)%2 != 0):
            data_x = data_x[:-1]
            data_y = data_y[:-1]
        self.data_x = data_x
        self.data_y = data_y
        train_x,test_x = self.split_data(data_x, split)
        train_y,test_y = self.split_data(data_y, split)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        filename = folder+stock_code+'.sav'
        self.stock_code = stock_code
        self.filename = filename
        clf = self.get_model(train_x,train_y,test_x,test_y,filename,times)
        self.model = clf
        score = clf.score(test_x,test_y)
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
    def build_decision_tree(self, x,y):
        decision_tree=DecisionTreeClassifier()
        decision_tree.fit(x,y)
        return decision_tree
    
    # get score by data
    def get_score_by_data(self, data):
        clf=self.build_decision_tree(data)
        x,y=self.split_feature_class(data)
        score = clf.score(x,y)
        return score

    # prediction
    def predict(self,data):
        result = self.model.predict(data)
        return result

    # predictions by different feature
    def sell_buy_different_day(self,data,sell,buy,days):
        result = []
        sum = 0
        predictions = self.predict(data)
        for i in range(0, predictions.size - 1):
            if ("yes".__eq__(predictions.item(i))):
                tmp = data[sell][i + days] - data[buy][i]
                sum = sum + tmp
                result.append(sum)
        return result

    # get_optimal_model
    def get_optimal_model(self,times,x,y,test_x,test_y):
        max_score = 0
        optimal_model = object()
        for i in range(0,times):
            model=self.build_decision_tree(x, y)
            score=model.score(test_x,test_y)
            if( score > max_score ):
                optimal_model = model
                max_score = score
        return optimal_model

    # build Decision Tree Model
    def get_model(self,x,y,test_x,test_y, filename,times):
        file = Path(filename)
        if file.is_file():
            if date.fromtimestamp(file_handler.creation_date(file)) == date.today():  # the date of records is not
                model=pickle.load(open(filename,'rb'))
            else:
                model = self.build_decision_tree(x, y)
                file_handler.dump_file(filename,model)
        else:
            # model = self.build_decision_tree(x, y)
            model = self.get_optimal_model(times,x,y, test_x, test_y)
            file_handler.dump_file(filename,model)
        return model


# load and prepare data
import stock_data,data_handler
import pandas as pd
stock_code = '300119'
data = stock_data.stock_data(stock_code).data
data = data.reindex(index=data.index[::-1])
y = pd.DataFrame(data_handler.get_class(data),columns=['class'])
dt = decision_tree_model(data,y,stock_code)
print(dt.score)

print("------------buy on next day but sell on the day after------")
print(dt.sell_buy_different_day(dt.test_x,'open','open',1))