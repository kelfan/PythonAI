#coding=utf-8
from math import log
import operator

#建立划分决策树的数据集，前两个值为特征是否存在，最后一个为键值，这里表示是否为鱼，相当于一个拥有两个特征的二分类问题
def createDataSet():
    dataSet = [[1, 1, 'yes'],             
               [1, 0, 'no'],
               [1, 1, 'yes'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels   #返回数据集合标签

#计算数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)         #数据集的长度，这里返回的结果是5
    labelCounts={}                  #创立一个数据字典，健为标签值，即yes或者no，键值为统计得到的次数，以此计算概率
    for featVec in dataSet:
        currentLabel=featVec[-1]       #最后一个表示数据的标签，这里即为yes或者no
        #print "featVec is : ",featVec  #每一次循环featVec的值都是列表中的一个子列表
        #print "currentLabel is : ",currentLabel   #currentLabel的值为数据标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel] += 1   #判断标签是否在当前字典的键值中，是的话，键值加1，不是的话，添加新的键到字典中
        #print "labelCounts is : ",labelCounts
        #print "labelCounts.keys is : ",labelCounts.keys()
        #print "labelCounts[currentLabel is] : ",labelCounts[currentLabel]
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries   #计算相应标签的概率
        shannonEnt -= prob *log(prob,2)           #计算信息熵并且返回
    return shannonEnt

#这个函数将数据集中的特种抽取出来，得到一个子数据集，即将数据集中的第axis个特征值为value的数据从数据集中抽取出来，同时在新得到的数据集中去掉这个特征
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet  #返回子数据集

#计算最好的抽取特征，这是决策树一个重要的树的和叶子的划分方式，有很多方法，相应的策略也不一样，这里是最简单的
#由于数据量小，直接用抽取得到的数据特征的信息熵的
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1              #数据中最后一个是标签，特征数量-1
    baseEntropy=calcShannonEnt(dataSet)
    #print "baseEntropy is :",baseEntropy
    bestInfoGain=0.0;bestFeature=-1            #初始化
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet] #抽取数据特种中的第i个特征组成特征列表
        #print "featList is :",featList
        uniqueVals=set(featList)
    #   print "uniqueVals is :",uniqueVals
        newEntropy=0.0
        for value in uniqueVals:                     #从每行中去掉上边抽取的相应第i个特征，计算信息熵
            subDateSet=splitDataSet(dataSet,i,value)
            prob=len(subDateSet)/float(len(dataSet))
            newEntropy += prob *calcShannonEnt(subDateSet)
        infoGain=baseEntropy-newEntropy
        if (infoGain >bestInfoGain):
            bestInfoGain =infoGain                   #计算得到最大的信息熵，并将选择的最好的特征划分返回
            bestFeature=i 
    return bestFeature

#决策树出现叶子节点由于每次划分是特征数量并不一定是减少的，当处理完了所有属性，类别标签仍然不是唯一的时候，需要人为定义叶子节点
#这个时候可以采用一些策略，这里采用出现次数最后的特征作为分类名称划分叶子节点
def majorityCnt(classList)  :
    classCount={}   #存储的是字典中每个对象出现的频率
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote] += 1
    #指定classCount.iteritems()的第二个数值进行降序排序，classCount.iteritems()返回的是字典类型，第二个域即为键值
    sortedClassCount=sort(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]   #返回出现次数最多的分类名称

#决策树的核心函数，建立决策树，用一个类似迭代的字典形式进行存储
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #两种中止策略，当类别完全相同，即信息熵已然是最大的，停止划分，或者当没有新的特征时，返回出现次数最多的特征
    if classList.count(classList[0]) == len(classList): 
        return classList[0]        #stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:       #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example [bestFeat] for example in dataSet]
    #print featValues
    uniqueVals = set(featValues)
    #print uniqueVals
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)  #嵌套迭代，获得最终的字典
    return myTree

#利用决策树进行分类
def classify(inputTree,featLabels,testVec):
    firstStr=inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else : classLabel = secondDict[key]
    return classLabel

#存储决策树
def storeTree(inputTree,filename) :
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

#获取决策树
def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)

# 以上是构建完了决策树，在命令框中输入以下代码可以得到返回的决策树：
#import tree
#mydat,labels=tree.createDataSet()
#myTree=tree.createTree(mydat)
#print(myTree)

