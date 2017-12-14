# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:32:04 2017

@author: Administrator
@title: Intro and Getting Stock Price Data - Python Programming for Finance
@Url: https://www.youtube.com/watch?v=2BrpKpWwT2A
"""

import datetime as dt 
import matplotlib.pyplot as plt 
from matplotlib import style 
import pandas as pd 
import pandas_datareader.data as web 

style.use('ggplot')

#@fan get data between start time and end time
#start = dt.datetime(2000,1,1)
#end = dt.datetime(2016,12,31) 

#@fan get TSLA's stock data from yahoo between start and end 
#df = web.DataReader('TSLA', 'yahoo', start, end)

#@fan print the head of the data 
#print(df.head())
#@fan save data into csv File 
#df.to_csv('tsla.csv')

#read stock data from csv file 
df  = pd.read_csv('tsla.csv',
                  parse_dates = True,
                  index_col=0)

#print(df.head())

#@fan draw the Diagram of The Data 
#df.plot()
#plt.show()

#@fan print one Column of the data 
df['Adj Close'].plot()
plt.show()

print(df[['Open', 'High']].head())