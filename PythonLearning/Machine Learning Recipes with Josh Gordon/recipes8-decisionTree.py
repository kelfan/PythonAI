# for python2/3 compatability
from __future__ import print_function

# 1颜色,2diameter列为Feature, 3为class种类 
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

# Column labels.
# These are used only to print the tree.
header = ["color", "diameter", "label"]

# 通过set无重复集合取出数据中某一列的所有唯一的数值
def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


#######
# Demo:
unique_vals(training_data, 0)
unique_vals(training_data, 1)
#######

# 统计数组中每个类型Label的个数 
def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

#######
# Demo:
class_counts(training_data)
#######

# 判断是不是数字类型 
def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

#######
# Demo:
is_numeric(7)
is_numeric("Red")
#######


class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """
    # 构建问题是把列和值传入
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # 根据之前问题定义的列和传入数组的列进行对比 -> 返回true/false
    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        # 如果是数字就判断大小,不是就判断相等
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    # 提示信息
    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

#######
# Demo:
# Let's write a question for a numeric attribute
Question(1, 3)
# How about one for a categorical attribute
q = Question(0, 'Green')
# Let's pick an example from the training set...
example = training_data[0]
# ... and see if it matches the question
q.match(example)
#######

# 把数组拆分为符合问题和不符合问题的两组数据 
# 结果 -> 返回 true_row, false_row 对的和错的行
def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


#######
# Demo:
# Let's partition the training data based on whether rows are Red.
true_rows, false_rows = partition(training_data, Question(0, 'Red'))
# This will contain all the 'Red' rows.
true_rows
# This will contain everything else.
false_rows
#######

# 算出结果有没混有别的可能 -> 返回0到1之间一个小数
def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    # 算出每种类型的总数  
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        # 算出概率为每种类型的数除以总数
        prob_of_lbl = counts[lbl] / float(len(rows))
        # 然后减去概率的平方
        impurity -= prob_of_lbl**2
    return impurity


#######
# Demo:
# Let's look at some example to understand how Gini Impurity works.
#
# First, we'll look at a dataset with no mixing.
no_mixing = [['Apple'],
             ['Apple']]
# this will return 0
gini(no_mixing)
#
# Now, we'll look at dataset with a 50:50 apples:oranges ratio
some_mixing = [['Apple'],
              ['Orange']]
# this will return 0.5 - meaning, there's a 50% chance of misclassifying
# a random example we draw from the dataset.
gini(some_mixing)
#
# Now, we'll look at a dataset with many different labels
lots_of_mixing = [['Apple'],
                 ['Orange'],
                 ['Grape'],
                 ['Grapefruit'],
                 ['Blueberry']]
# This will return 0.8
gini(lots_of_mixing)
#######

# info_gain 是 当前分离的比重
# @left 左边为对的行 true_row
# @right 右边为错的行 false_row
# @current_uncertainty 是gini算出的纯度
def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

#######
# Demo:
# Calculate the uncertainy of our training data.
current_uncertainty = gini(training_data)
#
# How much information do we gain by partioning on 'Green'?
true_rows, false_rows = partition(training_data, Question(0, 'Green'))
info_gain(true_rows, false_rows, current_uncertainty)
#
# What about if we partioned on 'Red' instead?
true_rows, false_rows = partition(training_data, Question(0,'Red'))
info_gain(true_rows, false_rows, current_uncertainty)
#
# It looks like we learned more using 'Red' (0.37), than 'Green' (0.14).
# Why? Look at the different splits that result, and see which one
# looks more 'unmixed' to you.
true_rows, false_rows = partition(training_data, Question(0,'Red'))
#
# Here, the true_rows contain only 'Grapes'.
true_rows
#
# And the false rows contain two types of fruit. Not too bad.
false_rows
#
# On the other hand, partitioning by Green doesn't help so much.
true_rows, false_rows = partition(training_data, Question(0,'Green'))
#
# We've isolated one apple in the true rows.
true_rows
#
# But, the false-rows are badly mixed up.
false_rows
#######

# 比对每个问题的info_gain的大小求最大的值 -> 返回问题和比重
# @rows 是数据集
def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns 测试数据是3列,这里结果是2

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

#######
# Demo:
# Find the best question to ask first for our toy dataset.
best_gain, best_question = find_best_split(training_data)
# FYI: is color == Red is just as good. See the note in the code above
# where I used '>='.
#######

# 模型: 每个枝叶记录预测每个类型的数量 
class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)

# 模型: 树上的每个节点都记载 
# @question 问题
# @ true_branch 一个decision_node 或者 leaf
# @ false_branch 一个decision_node 或者 leaf 
class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# 把数组分割为两部分,然后每部分再不断递归,最后 -> 输出node和leaf
def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    # 返回值得个数
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # 自身递归
    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # 自身递归
    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    # 返回 (问题, node节点1, node节点2) 
    return Decision_Node(question, true_branch, false_branch)


# 打印树形如下 
# Is diameter >= 3?
# --> True:
#   Is color == Yellow?
#   --> True:
#     Predict {'Apple': 1, 'Lemon': 1}
#   --> False:
#     Predict {'Apple': 1}
# --> False:
#   Predict {'Grape': 2}
def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    # 如果 node 是 leaf 类型
    if isinstance(node, Leaf):
        # 例如  Predict {'Apple': 1, 'Lemon': 1}
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    # 先继续打印 true_branch的分支, 递归 
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    # 再递归打印false那边的分支
    print_tree(node.false_branch, spacing + "  ")

# 放入的数据row是在树上的那个结果上 -> 输出预测的结果 
def classify(row, node):
    """See the 'rules of recursion' above."""

    # 如果没有分支就输出预测的结果 
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    # 判定跟分支的问题得到结果true/false -> 进入不同分支再判断
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


#######
# Demo:
# The tree predicts the 1st row of our
# training data is an apple with confidence 1.
my_tree = build_tree(training_data)
classify(training_data[0], my_tree)
#######

# 把所有的小数变成百分数 {'Apple': 1} -> {'Apple': '100%'} 
def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


#######
# Demo:
# Printing that a bit nicer {'Apple': 1} -> {'Apple': '100%'}
print_leaf(classify(training_data[0], my_tree))
#######

#######
# Demo:
# On the second example, the confidence is lower
print_leaf(classify(training_data[1], my_tree))
#######

if __name__ == '__main__':

    my_tree = build_tree(training_data)

    print_tree(my_tree)

    # Evaluate
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting
# - add support for regression


# output 
# Is diameter >= 3?
# --> True:
#   Is color == Yellow?
#   --> True:
#     Predict {'Apple': 1, 'Lemon': 1}
#   --> False:
#     Predict {'Apple': 1}
# --> False:
#   Predict {'Grape': 2}
# Actual: Apple. Predicted: {'Apple': '100%'}
# Actual: Apple. Predicted: {'Apple': '50%', 'Lemon': '50%'}
# Actual: Grape. Predicted: {'Grape': '100%'}
# Actual: Grape. Predicted: {'Grape': '100%'}
# Actual: Lemon. Predicted: {'Apple': '50%', 'Lemon': '50%'}