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

start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31) 

df = web.DataReader('TSLA', 'yahoo', start, end)
print(df.head())
