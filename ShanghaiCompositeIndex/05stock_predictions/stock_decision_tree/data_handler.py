import pandas as pd
def compare_features(sell,buy,data):
    result = []
    for i in range(0, len(data.index) - 1):
        tmp = data[sell][i + 1] - data[buy][i]
        if (tmp > 0):
            result.append('yes')
        else:
            result.append('no')
    count = result.count('yes')
    return result,count

def compare_method(data,feature,*args):
    result = []
    count = 0
    features = ''
    for i in args:
        tmp_result,tmp_count = compare_features(feature,i,data)
        if(tmp_count>count):
            result = tmp_result
            count = tmp_count
            features = i
    return result, count, features

def get_class(data):
    result,count, feature = compare_method(data,'low','high','open','close')
    return pd.DataFrame(list(result),columns=['class'])
