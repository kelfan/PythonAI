# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:32:04 2017

@author: Administrator
@title: Intro and Getting Stock Price Data - Python Programming for Finance
@Url: https://www.youtube.com/watch?v=2BrpKpWwT2A&list=PLQVvvaa0QuDcOdF96TBtRtuQksErCEBYZ
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

#@fan 从第100个开始,求平均值 Start from 100 and calculate the mean 
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0),rowspan=5,colspan=1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()

#@fan 放弃前面100个非数字的行 cut the rows that are not a Number(na)
df.dropna(inplace=True)

