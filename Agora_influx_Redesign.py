#!/usr/bin/env python
# coding: utf-8

# # 异常数据监测

# In[15]:


def load_db(host,port,username,password,database):
    from influxdb import InfluxDBClient
    client = InfluxDBClient(host,port,username,password,database)
    return client
client = load_db(host = '222.187.0.146', port = 8086, username = 'devops', password='agoradevops2018',database='quality_report')


# # 长期数据Time Series

# ## 定义

# In[7]:


##############################################################
## 读取包
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
###############################################################
## 定义 读取数据
def data_exp(df):
    import pandas as pd
    import numpy as np
    df['rate'] = df['succ']/df['total']
    ##Fill na with interpolate 插值了一下
    np.sum(np.isnan(df['succ']))
    #df = df.interpolate()
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.tz_convert(None)
    df['rate'] = (df['succ']/df['total']) * 100
#     df.plot(x = 'time', y ='succ')
#     df.plot(x = 'time', y = 'total')
#     df.plot(x ='time',y = 'rate' )
    df['savgol'] = signal.savgol_filter(df['rate'],9, # window size used for filtering
                       3) # order of fitted polynomial

###############################################################
## 定义Histogram Plot
def plot_hist(df):
    df['rate'] = df['succ']/df['total']
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.tz_convert(None)
    df.plot(x ='time',y = 'rate' )
    import matplotlib.pyplot as plt
    total = df['total']
    rate = df['rate']
    gridsize = (3, 2)
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (2, 0))
    ax3 = plt.subplot2grid(gridsize, (2, 1))
    ax1.set_title('Total usage against successful rate',
                  fontsize=14)
    sctr = ax1.scatter(x=total, y=rate, c=rate, cmap='RdYlGn')
    plt.colorbar(sctr, ax=ax1)
    ax2.hist(rate, bins='auto')
    ax3.hist(total, bins='auto', log=True)

    def add_titlebox(ax, text):
        ax.text(.55, .8, text,
                horizontalalignment='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.6),
                fontsize=12.5)
        return ax
    add_titlebox(ax2, 'Histogram: Successful Rate')
    add_titlebox(ax3, 'Histogram: Total (log scl.)')
    

###############################################################
# 定义拟合放好的模型
def LSTM_plot(df1,df2):
    import numpy
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset

    dataset = pd.DataFrame(df1['savgol'])
    dataset = dataset.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=10, verbose=1)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate prediction confidence interval
    train_upper = trainPredict + 2 * np.std(trainPredict)
    train_lower = trainPredict - 2 * np.std(trainPredict)
    test_upper = testPredict + 2 * np.std(testPredict)
    test_lower = testPredict - 2 * np.std(testPredict)
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#     # shift test predictions for plotting
#     testPredictPlot = numpy.empty_like(dataset)
#     testPredictPlot[:, :] = numpy.nan
#     testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#     # plot baseline and predictions
#     plt.plot(scaler.inverse_transform(dataset))
#     plt.plot(trainPredictPlot)
#     plt.plot(testPredictPlot)
#     plt.fill_between(np.arange(len(train_lower)),[float(i) for i in train_lower],[float(i) for i in train_upper],facecolor = 'purple',alpha = 0.2)
#     plt.fill_between(np.array([i for i in range(len(train_lower)+1,len(train_lower)+len(test_lower)+1)]),[float(i) for i in test_lower],[float(i) for i in test_upper],facecolor = 'blue',alpha = 0.2)
#     plt.show()
    
    dataset = pd.DataFrame(df2['rate'])
    dataset = dataset.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # reshape into X=t and Y=t+1
    look_back = 1
    fullX, fullY = create_dataset(dataset, look_back)
    # reshape input to be [samples, time steps, features]
    fullX = numpy.reshape(fullX, (fullX.shape[0], 1, fullX.shape[1]))
    fullPredict = model.predict(fullX)
    fullPredict = scaler.inverse_transform(fullPredict)
    fullY = scaler.inverse_transform([fullY])
    # calculate prediction confidence interval
    full_upper = fullPredict + 2 * np.std(fullPredict)
    full_lower = fullPredict - 2.5
    # calculate root mean squared error
    fullScore = math.sqrt(mean_squared_error(fullY[0], fullPredict[:,0]))
    print('full Score: %.5f RMSE' % (fullScore))
#     # shift train predictions for plotting
#     fullPredictPlot = numpy.empty_like(dataset)
#     fullPredictPlot[:, :] = numpy.nan
#     fullPredictPlot[look_back:len(fullPredict)+look_back, :] = fullPredict
#     # plot baseline and predictions
#     plt.plot(scaler.inverse_transform(dataset))
#     plt.plot(fullPredictPlot)
#     plt.fill_between(np.arange(len(full_lower)),[float(i) for i in full_lower],[float(i) for i in full_upper],facecolor = 'purple',alpha = 0.2)
#     plt.show()
    return fullY,fullPredict

###############################################################
# 展示异常点
def show_anomaly(df1,fullY,fullPredict):
    #Denote all anomaly points
    full_lower = fullPredict - 2.5
    full_array = np.append([False],fullY[0]<full_lower[:,0])
    full_array = np.append(full_array,[False])
    df1_ad = df1.loc[full_array,['time','rate','total']]
    print(df1_ad)

    mean_df1 = pd.pivot_table(df1, values='rate', index='time', aggfunc='mean')
    mean_df1['Per Three Hour'] = mean_df1['rate'].rolling(3).mean()
    mean_df1['rate'].plot(figsize=(20,6))
    plt.title('Last Period Avg Rate')
    plt.xlabel('Day')
    plt.ylabel('Rate')
    #Annotate large anomaly with red, and small with gray
    for index, row in df1_ad.loc[df1_ad['total']>=500,['time','rate']].iterrows():
        plt.annotate('Anomaly',xy=(row['time'],row['rate']),xytext = (row['time'],row['rate']-0.2),arrowprops=dict(facecolor='red',shrink = 0.05),fontsize = 15)
    for index, row in df1_ad.loc[df1_ad['total']<500,['time','rate']].iterrows():
        plt.annotate('Anomaly',xy=(row['time'],row['rate']),xytext = (row['time'],row['rate']-0.2),arrowprops=dict(facecolor='gray',shrink = 0.05),fontsize = 15)
    plt.show()


# In[13]:


import pandas as pd
import numpy as np

hangzhou_norm= pd.DataFrame(client.query("SELECT sum(success) as succ, sum(total) as total FROM edu_p2s WHERE idc = \'hangzhou9-cmcc\' AND time >= '2019-11-13T00:00:00Z'and time <= '2019-11-20T00:00:00Z'GROUP BY time(1m), idc fill(none)").get_points())
hangzhou_ab = pd.DataFrame(client.query("SELECT sum(success) as succ, sum(total) as total FROM edu_p2s WHERE idc = \'hangzhou9-cmcc\' AND time >= '2019-11-20T00:00:00Z'and time <= '2019-11-24T00:00:00Z'GROUP BY time(1m), idc fill(none)").get_points())
hangzhou = pd.concat([hangzhou_norm,hangzhou_ab])
plot_hist(hangzhou)
data_exp(hangzhou_norm)
data_exp(hangzhou_ab)


# In[42]:


fullY,fullPredict = LSTM_plot(hangzhou_norm,hangzhou_ab)
show_anomaly(hangzhou_ab,fullY,fullPredict)


# # 所有机房本周（12.03-12.10）数据 

# In[2]:


import pandas as pd
china_all = pd.read_csv("data_all.csv")
china_all.loc[china_all['totalavg'] > 144000,]


# In[3]:


china_all.loc[5,'idc']


# In[41]:


string = 'hangzhou9-cmcc'

exec("""df_norm = pd.DataFrame(client.query("SELECT sum(success) as succ, sum(total) as total FROM edu_p2s WHERE idc = \'%s\' AND time >= (now() - 6d - 12h) and time <= (now()- 1d) GROUP BY time(1m), idc fill(none)").get_points())"""%string)
print(df_norm)


# In[81]:


def anomaly_detection(string):
    train = "SELECT sum(success) as succ, sum(total) as total FROM edu_p2s WHERE idc = \'%s\' AND time >= '2019-11-14T00:00:00Z'and time <= '2019-11-20T00:00:00Z' GROUP BY time(1m), idc fill(none)"%string
    test = "SELECT sum(success) as succ, sum(total) as total FROM edu_p2s WHERE idc = \'%s\' AND time >= '2019-11-20T00:00:00Z'and time <= '2019-11-24T00:00:00Z' GROUP BY time(1m), idc fill(none)"%string
    df_train = pd.DataFrame(client.query(train).get_points())
    df_test = pd.DataFrame(client.query(test).get_points())
    if df_train.empty or df_test.empty:
        return print("Data missing, no operation needed")
    else:
        data_exp(df_train)
        data_exp(df_test)
        fullY,fullPredict = LSTM_plot(df_train,df_test)
        show_anomaly(df_test,fullY,fullPredict)
    
anomaly_detection('hangzhou9-cmcc')


# In[4]:


def anomaly_detection_new(string):
    train = "SELECT sum(success) as succ, sum(total) as total FROM edu_p2s WHERE idc = \'%s\' AND time >= (now() - 7d -12h) and time <= (now()- 2d) GROUP BY time(1m), idc fill(none)"%string
    test = "SELECT sum(success) as succ, sum(total) as total FROM edu_p2s WHERE idc = \'%s\' AND time >= (now() - 2d) and time <= (now() - 5m) GROUP BY time(1m), idc fill(none)"%string
    df_train = pd.DataFrame(client.query(train).get_points())
    df_test = pd.DataFrame(client.query(test).get_points())
    if df_train.empty or df_test.empty:
        return print("Data missing, no operation needed")
    else:
        data_exp(df_train)
        data_exp(df_test)
        fullY,fullPredict = LSTM_plot(df_train,df_test)
        show_anomaly(df_test,fullY,fullPredict)


# In[8]:


for index,row in china_all.iterrows():
    anomaly_detection_new(row['idc'])
    if index == 10:
        break

