# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:32:04 2017

@author: kelfan
@source: [Decision Tree - python implementation](https://www.youtube.com/watch?v=GE2P2DlIj9k)
"""
def createDataSet():
	dataSet = [[0,1,1,'yes'],
	[0,1,0,'no'],
	[1,1,0,'no'],
	[0,1,1,'no'],
	[1,1,1,'no'],
	[1,0,0,'no'],
	[0,0,1,'no'],
	[1,0,1,'no'],
	[1,1,0,'no'],
	[0,0,0,'no'],
	[0,1,1,'no'],]
	labels = ['cartoon', 'winter', 'more than 1 person']
	return dataSet, labels

def createTree(dataSet, labels):
	# extracting data
	classlist = [example[-1] for example in dataSet]
	if classlist.count(classlist[0]) == len(classlist):
		return classlist[0] # stop splitting when all of the classes are equal
	if len(dataSet[0]) ==1: # stop splitting when there are no more features in dataSet
		return majorityCnt(classlist)
	# use information gain
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	# build a Tree recursively
	myTree = {bestFeatLabel: {}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:] # copy all of labels, so tree don't mess up existing labels
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value), subLabels)
	return myTree

# entropy
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet: # the number of unique elements and their occurance
		currentLabel = featVec[-1] 
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0 
		labelCounts[currentLabel] += 1 
	ShannonEnt = 0.0 
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		ShannonEnt -= prob * log(prob, 2) # log base 2 
	return ShannonEnt

# classifier 
def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	key = testVec[featIndex]
	valueOfFeat = secondDict[key]
	if isinstance(valueOfFeat, dict):
		classLabel = classify(valueOfFeat, featLabels, testVec)
	else:
		classLabel = valueOfFeat
	return classLabel



# collect Data
myData, labels = createDataSet()

# build a tree
mytree = createTree(myData, labels)
print(mytree)

print("give me any photo")

#run test
answer = classify(mytree, ['cartoon','winter','more than 1 person'],[0,1,1])
print("hi, answer is "+ answer)

answer = classify(mytree, ['cartoon','winter','more than 1 person'],[1,1,1])
print("hi, answer is "+answer )
