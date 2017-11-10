# LSTM for international airline passengers problem with regression framing
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

# 导入相关包
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 固定随机数,从而可以重复实验
# fix random seed for reproducibility
numpy.random.seed(7)


"""
# 导入数据
# load the dataset
表中数据如下
	"Month","International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60"
	"1949-01",112
	"1949-02",118
	"1949-03",132
	"1949-04",129
	"1949-05",121
usecols=[1] 
	是第二列数据(112,118...),因为日期没用,每个数据间隔一个月,所以直接去掉, 
	usecols=[0]就是第一列
skipfooter=3 
	表示最后3行footer lines不要<=因为最后有句没用的数据; 
	1 的话会最后多个nan,2和3结果一样; 4的话就会少了最后的数据;
"""
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

"""
# 把数据变成0-1之间的小数
# normalize the dataset
MinMaxScaler
	sciki-learn 库
	把数据变成0-1之间的小数
"""
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

"""
# 把数据变成训练集和测试集
# split into train and test sets
train_size = int(len(dataset) * 0.67)
	把数据集的67%作为训练集
	剩余的作为测试集
dataset[0:train_size,:]
	0到train_size行,所有列
dataset[train_size:len(dataset),:]
	train_size行到最后行,所有列
"""
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

"""
# convert an array of values into a dataset matrix
dataset 
	输入的数据集
	a NumPy array that we want to convert into a dataset
look_back 
	每次跳的跨度,默认1
	the number of previous time steps to use as input variables to predict the next time period 
获取的输出: 
	X=t and Y=t+1
	后面的数字是前面的预测结果,是后面的输入  
	X		Y
	112		118
	118		132
	132		129
	129		121
	121		135
"""
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		# 每次获取第i行数据 第1列 index=0 look_back如果取别的数就是取多个数作为数组
		a = dataset[i:(i+look_back), 0]
		# 返回类型不一样 <class 'numpy.ndarray'>
		print(a)
		# 返回类型不一样 <class 'numpy.float32'>
		print(dataset[i,0])
		dataX.append(a)
		# 获取 i+1 行 index0 列数据
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# 获取训练集和测试集的 测试数据和预测结果
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

"""
# 把数组的形式由 [samples, features] 变成 LSTM 接受的形式 [samples, time steps, features]
# reshape input to be [samples, time steps, features]
reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	把trainX的数据由二维变成了三维(行,列,高)
	这里是94行1列1高
	格式 reshape(数据组,(行,列,高...))
trainX.shape[0]
	是 trainX 的行数
trainX.shape[1]
	是 trainX 的列数
"""
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

"""
# 创建神经网络 
# create and fit the LSTM network
"""
model = Sequential()
# 增加LSTM输入层: 输入1和look_back, 输出4个
model.add(LSTM(4, input_shape=(1, look_back)))
# 输出预测1个
model.add(Dense(1))
# loss 损失函数 optimizer 优化函数
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型: 
# epochs训练100回,
# batch_size每回放入1个数据, 
# verbos显示过程
# 	[0是不显示,1是进度条,2是数字,3以上是只有数数]
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 加载训练过的结果
model.load_weights('./weights')

"""
# 进行预测
# make predictions
"""
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

"""
# invert predictions
inverse_transform()
	把normalized的0-1之间的小数还原为100+的正常数字
"""
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

"""
# 打印测试结果
mean_squared_error 是损失函数
	损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。
sqrt 
	求平方根
trainY[0]
	第一行数据
	[ 117.99999916  131.99999879  129.00000163 ..., 270.99999355]
"""
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# Train Score: 22.92 RMSE
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# Test Score: 47.53 RMSE
print('Test Score: %.2f RMSE' % (testScore))




"""
可视化显示结果
numpy.empty_like 
	创建一个类似的数据
	Return a new array with the same shape and type as a given array
numpy.nan 
	把数据变成为空集
	[[ nan]
	 ..., 
	 [ nan]]
最后一句
	把trainPredict的数据填入
	trainPredictPlot
		look_back开始到训练数据的长度+look_back是训练预测的数据
	testPredictPlot 
		train数据长度+两个lookback+1 到 数据集的长度-1
"""
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# 分别描绘三条线
# plot baseline and predictions
# inverse_transform 把normalize的数据还原为原来的数据
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# 保存训练过的结果
model.save_weights('./weights')
