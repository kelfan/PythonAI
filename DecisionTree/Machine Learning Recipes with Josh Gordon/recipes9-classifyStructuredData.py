# -*- coding: utf-8 -*-
"""
Intro to Classifying Structured Data with TensorFlow
https://github.com/random-forests/tensorflow-workshop/blob/master/examples/07_structured_data.ipynb?utm_campaign=ai_series_tensorflowcode_103017&utm_source=gdev&utm_medium=yt-desc
Created on Sun Dec 10 22:52:30 2017

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd

from IPython.display import Image

import tensorflow as tf
print('This code requires TensorFlow v1.3+')
print('You have:', tf.__version__)
# This code requires TensorFlow v1.3+
# You have: 1.3.0

"""
dataset
"""
census_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
census_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
census_train_path = tf.contrib.keras.utils.get_file('census.train', census_train_url)
census_test_path = tf.contrib.keras.utils.get_file('census.test', census_test_url)

column_names = [
  'age', 'workclass', 'fnlwgt', 'education', 'education-num',
  'marital-status', 'occupation', 'relationship', 'race', 'gender',
  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
  'income'
]

# Load the using Pandas

# Notes
# 1) We provide the header from above.
# 2) The test file has a line we want to disgard at the top, so we include the parameter 'skiprows=1'
census_train = pd.read_csv(census_train_path, index_col=False, names=column_names) 
census_test = pd.read_csv(census_test_path, skiprows=1, index_col=False, names=column_names) 

# Drop any rows that have missing elements
# Of course there are other ways to handle missing data, but we'll
# take the simplest approach here.
census_train = census_train.dropna(how="any", axis=0)
census_test = census_test.dropna(how="any", axis=0)

"""
Correct formatting problems with the Census data
@income 是字符串 取值 ">50k" "<=50k"
"""
# Separate the label we want to predict into its own object 
# At the same time, we'll convert it into true/false to fix the formatting error
census_train_label = census_train.pop('income').apply(lambda x: ">50K" in x) # 把income那行取出来作为新的变量,原来的数据中会减少这行
census_test_label = census_test.pop('income').apply(lambda x: ">50K" in x)


print ("Training examples: %d" % census_train.shape[0])
print ("Training labels: %d" % census_train_label.shape[0])
print()
print ("Test examples: %d" % census_test.shape[0])
print ("Test labels: %d" % census_test_label.shape[0])

census_train.head()
census_train_label.head(10)

"""
Input functions for training and testing data
"""
def create_train_input_fn(): 
    return tf.estimator.inputs.pandas_input_fn(
        x=census_train,
        y=census_train_label, 
        batch_size=32,
        num_epochs=None, # Repeat forever
        shuffle=True)

def create_test_input_fn():
    return tf.estimator.inputs.pandas_input_fn(
        x=census_test,
        y=census_test_label, 
        num_epochs=1, # Just one epoch
        shuffle=False) # Don't shuffle so we can compare to census_test_labels later
    
"""
Feature Engineering
@fan 把Feature转为bucket分段 hash cross 列出所有取值
"""
# A list of the feature columns we'll use to train the Linear model
feature_columns = []
# To start, we'll use the raw, numeric value of age.
age = tf.feature_column.numeric_column('age') # 关于age的元数据 
feature_columns.append(age) 

# @fan 两种分段的方式,下面是自定义的范围
age_buckets = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('age'), 
    boundaries=[31, 46, 60, 75, 90] # specify the ranges
)

feature_columns.append(age_buckets)

# @fan 两种分段的方式,下面是分成10段?
# You can also evenly divide the data, if you prefer not to specify the ranges yourself.
age_buckets = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('age'), 
    list(range(10))
)

# @fan 列出所有的可能值
# Here's a categorical column
# We're specifying the possible values
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])

feature_columns.append(education)

# @fan 进行hash取值 
# A categorical feature with a possibly large number of values
# and the vocabulary not specified in advance.
native_country = tf.feature_column.categorical_column_with_hash_bucket('native-country', 1000)
feature_columns.append(native_country)

# @fan 交叉特征 
age_cross_education = tf.feature_column.crossed_column(
    [age_buckets, education],
    hash_bucket_size=int(1e4) # Using a hash is handy here
)
feature_columns.append(age_cross_education)

"""
Train a Canned Linear Estimator
""" 
train_input_fn = create_train_input_fn()
estimator = tf.estimator.LinearClassifier(feature_columns, model_dir='graphs/linear', n_classes=2)
estimator.train(train_input_fn, steps=1000)

# evaluate 
test_input_fn = create_test_input_fn()
estimator.evaluate(test_input_fn)

# predict 
# reinitialize the input function
test_input_fn = create_test_input_fn()

predictions = estimator.predict(test_input_fn)
i = 0
for prediction in predictions:
    true_label = census_test_label[i]
    predicted_label = prediction['class_ids'][0]
    # Uncomment the following line to see probabilities for individual classes
    # print(prediction) 
    print("Example %d. Actual: %d, Predicted: %d" % (i, true_label, predicted_label))
    i += 1
    if i == 5: break











