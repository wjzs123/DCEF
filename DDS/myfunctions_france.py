import numpy
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from math import sqrt

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
def svr_model(new_data, i, look_back, data_partition, cap):
    import numpy as np
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()
    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    datasetss2 = pd.DataFrame(s)
    datasets = datasetss2.values

    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()
    import tensorflow as tf

    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    from sklearn.svm import SVR

    grid = SVR(kernel='rbf')
    grid.fit(X, y)
    y_pred_train_svr = grid.predict(X)
    y_pred_test_svr = grid.predict(X1)

    y_pred_train_svr = pd.DataFrame(y_pred_train_svr)
    y_pred_test_svr = pd.DataFrame(y_pred_test_svr)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_pred_test1_svr = sc_y.inverse_transform(y_pred_test_svr)
    y_pred_train1_svr = sc_y.inverse_transform(y_pred_train_svr)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    y_pred_test1_svr = pd.DataFrame(y_pred_test1_svr, columns=["Predicted_CO2"])
    y_test = pd.DataFrame(y_test, columns=["Actual_CO2"])

    # 保存到CSV文件
    result_df = pd.concat([y_test, y_pred_test1_svr], axis=1)
    result_df.to_csv('svr_results.csv', index=False)

    y_pred_test1_svr = np.array(y_pred_test1_svr)
    y_test = np.array(y_test)

    # print(f'Type of y_test: {type(y_test)}')
    # print(f'Type of y_pred_test1_svr: {type(y_pred_test1_svr)}')

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - y_pred_test1_svr)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred_test1_svr))
    mae = metrics.mean_absolute_error(y_test, y_pred_test1_svr)
    r2 = r2_score(y_test, y_pred_test1_svr)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def dt_model(new_data, i, look_back, data_partition, cap):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import metrics
    from math import sqrt

    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()
    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    datasetss2 = pd.DataFrame(s)
    datasets = datasetss2.values

    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    # 使用 create_dataset 函数生成 X 和 Y
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    # 数据标准化
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    import tensorflow as tf
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # 导入决策树回归模型
    from sklearn.tree import DecisionTreeRegressor

    # 使用决策树回归模型
    dt = DecisionTreeRegressor()
    dt.fit(X, y)

    y_pred_train_dt = dt.predict(X)
    y_pred_test_dt = dt.predict(X1)

    y_pred_train_dt = pd.DataFrame(y_pred_train_dt)
    y_pred_test_dt = pd.DataFrame(y_pred_test_dt)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    # 反标准化
    y_pred_test1_dt = sc_y.inverse_transform(y_pred_test_dt)
    y_pred_train1_dt = sc_y.inverse_transform(y_pred_train_dt)
    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    # 保存预测结果
    y_pred_test1_dt = pd.DataFrame(y_pred_test1_dt, columns=["Predicted_CO2"])
    y_test = pd.DataFrame(y_test, columns=["Actual_CO2"])

    result_df = pd.concat([y_test, y_pred_test1_dt], axis=1)
    result_df.to_csv('dt_results.csv', index=False)

    y_pred_test1_dt = np.array(y_pred_test1_dt)
    y_test = np.array(y_test)

    # 评估模型性能
    mape = np.mean((np.abs(y_test - y_pred_test1_dt)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred_test1_dt))
    mae = metrics.mean_absolute_error(y_test, y_pred_test1_dt)
    r2 = r2_score(y_test, y_pred_test1_dt)

    print('MAPE:', mape)
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('R-squared score:', r2)

def ann_model(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()
    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    datasetss2 = pd.DataFrame(s)
    datasets = datasetss2.values

    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()
    import tensorflow as tf

    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.models import Sequential
    from tensorflow.python.keras.layers.core import Dense, Dropout, Activation


    neuron = 128
    model = Sequential()
    model.add(Dense(units=neuron, input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(trainX1, y, verbose=0)

    # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test = numpy.array(y_pred_test).ravel()

    y_pred_test = pd.DataFrame(y_pred_test)
    y_pred_test1 = sc_y.inverse_transform(y_pred_test)
    y1 = pd.DataFrame(y1)

    y_test = sc_y.inverse_transform(y1)

    # 将 y_test 和 y_pred_test1 保存到 CSV 文件中
    y_pred_test1 = pd.DataFrame(y_pred_test1, columns=["Predicted_CO2"])
    y_test = pd.DataFrame(y_test, columns=["Actual_CO2"])

    # 合并实际值和预测值为一个 DataFrame
    result_df = pd.concat([y_test, y_pred_test1], axis=1)

    # 保存为 CSV 文件
    result_df.to_csv('ann_results.csv', index=False)

    y_pred_test1_svr = numpy.array(y_pred_test1)
    y_test = numpy.array(y_test)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - y_pred_test1)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred_test1))
    mae = metrics.mean_absolute_error(y_test, y_pred_test1)
    r2 = r2_score(y_test, y_pred_test1)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def rf_model(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    datasetss2 = pd.DataFrame(s)
    datasets = datasetss2.values

    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()
    import tensorflow as tf

    import numpy

    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    from sklearn.ensemble import RandomForestRegressor

    grid = RandomForestRegressor()
    grid.fit(X, y)
    y_pred_train_rf = grid.predict(X)
    y_pred_test_rf = grid.predict(X1)

    y_pred_train_rf = pd.DataFrame(y_pred_train_rf)
    y_pred_test_rf = pd.DataFrame(y_pred_test_rf)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_pred_test1_rf = sc_y.inverse_transform(y_pred_test_rf)
    y_pred_train1_rf = sc_y.inverse_transform(y_pred_train_rf)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    # y_pred_test1_rf=pd.DataFrame(y_pred_test1_rf)
    # y_pred_train1_rf=pd.DataFrame(y_pred_train1_rf)

    # y_test= pd.DataFrame(y_test)

    y_pred_test1_rf = pd.DataFrame(y_pred_test1_rf, columns=["Predicted_CO2"])
    # y_pred_train1_rf = pd.DataFrame(y_pred_train1_rf, columns=["Predicted_CO2"])

    y_test = pd.DataFrame(y_test, columns=["Actual_CO2"])

    # 保存到CSV文件
    result_df = pd.concat([y_test, y_pred_test1_rf], axis=1)
    result_df.to_csv('rf_results.csv', index=False)

    y_pred_test1_svr = numpy.array(y_pred_test1_rf)
    y_test = numpy.array(y_test)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - y_pred_test1_rf)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred_test1_rf))
    mae = metrics.mean_absolute_error(y_test, y_pred_test1_rf)
    r2 = r2_score(y_test, y_pred_test1_rf)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def lstm_model(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    datasetss2 = pd.DataFrame(s)
    datasets = datasetss2.values

    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()
    import tensorflow as tf

    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.models import Sequential
    from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
    from tensorflow.python.keras.layers.recurrent import LSTM

    neuron = 128
    model = Sequential()
    model.add(LSTM(units=neuron, input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(trainX1, y, epochs=100, batch_size=64, verbose=0)
    # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test = numpy.array(y_pred_test).ravel()

    y_pred_test = pd.DataFrame(y_pred_test)
    y_pred_test1 = sc_y.inverse_transform(y_pred_test)
    y1 = pd.DataFrame(y1)

    y_test = sc_y.inverse_transform(y1)

    # 将 y_test 和 y_pred_test1 保存到 CSV 文件中
    result_df = pd.DataFrame({
        'Actual_CO2': y_test.ravel(),
        'Predicted_CO2': y_pred_test1.ravel()
    })

    # 保存为 CSV 文件
    result_df.to_csv('lstm_results.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - y_pred_test1)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred_test1))
    mae = metrics.mean_absolute_error(y_test, y_pred_test1)
    r2 = r2_score(y_test, y_pred_test1)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def tcn_model(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    datasetss2 = pd.DataFrame(s)
    datasets = datasetss2.values

    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    import tensorflow as tf
    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    # TCN import
    from tcn import TCN
    from keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # Model building
    model = Sequential()
    model.add(TCN(input_shape=(trainX1.shape[1], trainX1.shape[2])))  # Replace LSTM with TCN
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(trainX1, y, epochs=100, batch_size=64, verbose=0)

    # Make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test = numpy.array(y_pred_test).ravel()

    y_pred_test = pd.DataFrame(y_pred_test)
    y_pred_test1 = sc_y.inverse_transform(y_pred_test)
    y1 = pd.DataFrame(y1)

    y_test = sc_y.inverse_transform(y1)

    # Save results to CSV
    result_df = pd.DataFrame({
        'Actual_CO2': y_test.ravel(),
        'Predicted_CO2': y_pred_test1.ravel()
    })
    result_df.to_csv('tcn_results.csv', index=False)

    # Summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - y_pred_test1)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred_test1))
    mae = metrics.mean_absolute_error(y_test, y_pred_test1)
    r2 = r2_score(y_test, y_pred_test1)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def bitcn_model(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    datasetss2 = pd.DataFrame(s)
    datasets = datasetss2.values

    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    import tensorflow as tf
    import numpy as np
    np.random.seed(1234)
    tf.random.set_seed(1234)

    trainX1 = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = np.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Dense, Flatten, BatchNormalization, Activation, Dropout
    from tensorflow.keras.optimizers import Adam

    neuron = 128
    kernel_size = 3

    # 构建TCN模型
    model = Sequential()
    model.add(Conv1D(filters=neuron, kernel_size=kernel_size, padding='causal',
                     input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=neuron, kernel_size=kernel_size, padding='causal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(trainX1, y, epochs=100, batch_size=64, verbose=0)

    # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test = np.array(y_pred_test).ravel()

    y_pred_test = pd.DataFrame(y_pred_test)
    y_pred_test1 = sc_y.inverse_transform(y_pred_test)
    y1 = pd.DataFrame(y1)

    y_test = sc_y.inverse_transform(y1)

    # 将 y_test 和 y_pred_test1 保存到 CSV 文件中
    result_df = pd.DataFrame({
        'Actual_CO2': y_test.ravel(),
        'Predicted_CO2': y_pred_test1.ravel()
    })

    # 保存为 CSV 文件
    result_df.to_csv('bitcn_results.csv', index=False)

    # summarize the fit of the model
    mape = np.mean((np.abs(y_test - y_pred_test1)) / y_test) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test1))
    mae = metrics.mean_absolute_error(y_test, y_pred_test1)
    r2 = r2_score(y_test, y_pred_test1)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def vmd_bitcn(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    from vmdpy import VMD

    # VMD parameters
    alpha = 2000  # bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 5  # number of modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7

    # Perform VMD
    imfs, u_hat, omega = VMD(s, alpha, tau, K, DC, init, tol)

    full_imf = pd.DataFrame(imfs)
    data_decomp = full_imf.T

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    neuron = 128
    lr = 0.001
    optimizer = 'Adam'

    for col in data_decomp:
        datasetss2 = pd.DataFrame(data_decomp[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        import numpy
        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
        from tensorflow.python.keras.layers.recurrent import LSTM

        neuron = 128
        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)
    # a = a.apply(lambda x: x[0])

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('vmd_lstm_results.csv', index=False)

    min_len = min(len(y_test), len(a))
    y_test_aligned = y_test.iloc[:min_len]
    a_aligned = a.iloc[:min_len]

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    # rmse = sqrt(mean_squared_error(y_test, a))
    rmse = sqrt(mean_squared_error(y_test_aligned, a_aligned))
    mae = metrics.mean_absolute_error(y_test_aligned, a_aligned)
    r2 = r2_score(y_test_aligned, a_aligned)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def emd_bitcn(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    from PyEMD import EMD
    import ewtpy

    emd = EMD()

    IMFs = emd(s)

    full_imf = pd.DataFrame(IMFs)
    data_decomp = full_imf.T

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    neuron = 128
    lr = 0.001
    optimizer = 'Adam'

    for col in data_decomp:
        datasetss2 = pd.DataFrame(data_decomp[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        import numpy
        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
        from tensorflow.python.keras.layers.recurrent import LSTM

        neuron = 128
        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)
    # a = a.apply(lambda x: x[0])

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('emd_lstm_results.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def eemd_bitcn(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    dfs = datas.values

    # 使用 EEMD 而不是 EMD
    from PyEMD import EEMD
    # eemd = EEMD()

    eemd = EEMD(noise_width=0.0788)
    eemd.noise_seed(12345)

    IMFs = eemd(dfs)

    full_imf = pd.DataFrame(IMFs)
    data_decomp = full_imf.T

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    neuron = 128
    lr = 0.001
    optimizer = 'Adam'

    for col in data_decomp:
        datasetss2 = pd.DataFrame(data_decomp[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        tf.random.set_seed(1234)

        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('eemd_lstm_results.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)


def ceemd_bitcn(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    from PyEMD import CEEMD

    emd = CEEMD(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf = pd.DataFrame(IMFs)
    data_decomp = full_imf.T

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    neuron = 128
    lr = 0.001
    optimizer = 'Adam'

    for col in data_decomp:
        datasetss2 = pd.DataFrame(data_decomp[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        import numpy
        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
        from tensorflow.python.keras.layers.recurrent import LSTM

        neuron = 128
        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('ceemd_lstm_results.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def ceemdan_bitcn(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    from PyEMD import CEEMDAN

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf = pd.DataFrame(IMFs)
    data_decomp = full_imf.T

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    neuron = 128
    lr = 0.001
    optimizer = 'Adam'

    for col in data_decomp:
        datasetss2 = pd.DataFrame(data_decomp[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        import numpy
        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
        from tensorflow.python.keras.layers.recurrent import LSTM

        neuron = 128
        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('ceemdan_lstm_results.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def ceemdan_vmd_bitcn(new_data, i, look_back, data_partition, cap):
    import pandas as pd
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    from PyEMD import CEEMDAN
    import numpy as np
    # import pandas as pd
    from vmdpy import VMD

    # 使用CEEMDAN分解信号
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    # 分解得到IMFs
    IMFs = emd(s)

    # 转换IMFs为DataFrame格式
    full_imf = pd.DataFrame(IMFs)
    ceemdan1 = full_imf.T

    # 选择第一个IMF
    imf1 = ceemdan1.iloc[:, 0]
    imf_dataps = np.array(imf1)
    imf_datasetss = imf_dataps.reshape(-1, 1)
    imf_new_datasets = pd.DataFrame(imf_datasetss)

    # 使用VMD分解第一个IMF
    alpha = 2000  # 调节模态的惩罚项
    tau = 0.  # 通过噪声影响的时间步数
    K = 3  # 模态数量
    DC = 0  # 无直流成分
    init = 1  # 初始化
    tol = 1e-7  # 容忍度

    # 执行VMD
    u, u_hat, omega = VMD(imf1, alpha, tau, K, DC, init, tol)

    # 将VMD结果转换为DataFrame格式
    df_vmd = pd.DataFrame(u.T)

    # 如果只想保留前两个模态（类似于EWT中drop掉第三个模态）
    df_vmd.drop(df_vmd.columns[2], axis=1, inplace=True)

    # 将保留的模态进行去噪处理，即将其求和
    denoised = df_vmd.sum(axis=1, skipna=True)

    # 将去噪后的数据与其他IMFs结合
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # new_ceemdan现在包含VMD去噪后的结果和其他IMFs

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    lr = 0.001
    optimizer = 'Adam'

    for col in new_ceemdan:
        datasetss2 = pd.DataFrame(new_ceemdan[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
        from tensorflow.python.keras.layers.recurrent import LSTM

        neuron = 128
        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('ceemdan_vmd_lstm_results.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)


def ceemdan_emd_bitcn(new_data, i, look_back, data_partition, cap):
    import pandas as pd
    import numpy as np
    from PyEMD import EMD, CEEMDAN
    # 提取数据
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True).dropna()

    datas = data1['co2']
    s = datas.values

    # 使用CEEMDAN分解信号
    emd_ceemdan = CEEMDAN(epsilon=0.05)
    emd_ceemdan.noise_seed(12345)
    IMFs = emd_ceemdan(s)

    full_imf = pd.DataFrame(IMFs)
    ceemdan1 = full_imf.T

    # 选择第一个IMF
    imf1 = ceemdan1.iloc[:, 0].values  # 使用.values 转换为 numpy 数组

    # 确保 imf1 是一维数组
    print(f"IMF1 shape: {imf1.shape}")

    # 使用EMD分解第一个IMF
    emd = EMD()
    try:
        IMFs_emd = emd(imf1)
    except Exception as e:
        print(f"Error during EMD decomposition: {e}")
        return

    # 将EMD结果转换为DataFrame格式
    df_emd = pd.DataFrame(IMFs_emd).T

    # 去噪处理
    denoised = df_emd.sum(axis=1, skipna=True)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # 打印数据的形状
    print(f"new_ceemdan shape: {new_ceemdan.shape}")

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 200
    batch_size = 64
    lr = 0.001
    optimizer = 'Adam'

    for col in new_ceemdan:
        datasetss2 = pd.DataFrame(new_ceemdan[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size

        if train_size <= 0 or test_size <= 0:
            raise ValueError("Invalid train_size or test_size. Ensure that data_partition is set correctly.")

        train, test = datasets[:train_size], datasets[train_size:]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.transform(X_test)
        y1 = sc_y.transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        trainX = np.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        np.random.seed(1234)
        tf.random.set_seed(1234)

        model = Sequential()
        model.add(LSTM(units=128, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        y_pred_test = np.array(y_pred_test).ravel()
        y_pred_train = np.array(y_pred_train).ravel()

        y_test = sc_y.inverse_transform(pd.DataFrame(y1))
        y_train = sc_y.inverse_transform(pd.DataFrame(y))
        y_pred_test1 = sc_y.inverse_transform(pd.DataFrame(y_pred_test))
        y_pred_train1 = sc_y.inverse_transform(pd.DataFrame(y_pred_train))

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(datas)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size

    if train_size <= 0 or test_size <= 0:
        raise ValueError("Invalid train_size or test_size. Ensure that data_partition is set correctly.")

    train, test = dataset[:train_size], dataset[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.transform(X_test)
    y1 = sc_y.transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = np.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    np.random.seed(1234)
    tf.random.set_seed(1234)

    y_test = sc_y.inverse_transform(pd.DataFrame(y1))
    y_train = sc_y.inverse_transform(pd.DataFrame(y))

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('ceemdan_emd_lstm_results.csv', index=False)

    mape = np.mean((np.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def ceemdan_eemd_bitcn(new_data, i, look_back, data_partition, cap):
    import pandas as pd
    import numpy as np
    from PyEMD import EEMD, CEEMDAN
    # 提取数据
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True).dropna()

    datas = data1['co2']
    s = datas.values

    # 使用CEEMDAN分解信号
    emd_ceemdan = CEEMDAN(epsilon=0.05)
    emd_ceemdan.noise_seed(12345)
    IMFs = emd_ceemdan(s)

    full_imf = pd.DataFrame(IMFs)
    ceemdan1 = full_imf.T

    # 选择第一个IMF
    imf1 = ceemdan1.iloc[:, 0].values  # 使用.values 转换为 numpy 数组

    # 确保 imf1 是一维数组
    print(f"IMF1 shape: {imf1.shape}")

    # 使用EMD分解第一个IMF
    eemd = EEMD()
    try:
        IMFs_emd = eemd(imf1)
    except Exception as e:
        print(f"Error during EMD decomposition: {e}")
        return

    # 将EMD结果转换为DataFrame格式
    df_emd = pd.DataFrame(IMFs_emd).T

    # 去噪处理
    denoised = df_emd.sum(axis=1, skipna=True)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # 打印数据的形状
    print(f"new_ceemdan shape: {new_ceemdan.shape}")

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    lr = 0.001
    optimizer = 'Adam'

    for col in new_ceemdan:
        datasetss2 = pd.DataFrame(new_ceemdan[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size

        if train_size <= 0 or test_size <= 0:
            raise ValueError("Invalid train_size or test_size. Ensure that data_partition is set correctly.")

        train, test = datasets[:train_size], datasets[train_size:]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.transform(X_test)
        y1 = sc_y.transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        trainX = np.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        np.random.seed(1234)
        tf.random.set_seed(1234)

        model = Sequential()
        model.add(LSTM(units=128, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        y_pred_test = np.array(y_pred_test).ravel()
        y_pred_train = np.array(y_pred_train).ravel()

        y_test = sc_y.inverse_transform(pd.DataFrame(y1))
        y_train = sc_y.inverse_transform(pd.DataFrame(y))
        y_pred_test1 = sc_y.inverse_transform(pd.DataFrame(y_pred_test))
        y_pred_train1 = sc_y.inverse_transform(pd.DataFrame(y_pred_train))

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(datas)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size

    if train_size <= 0 or test_size <= 0:
        raise ValueError("Invalid train_size or test_size. Ensure that data_partition is set correctly.")

    train, test = dataset[:train_size], dataset[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.transform(X_test)
    y1 = sc_y.transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = np.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    np.random.seed(1234)
    tf.random.set_seed(1234)

    y_test = sc_y.inverse_transform(pd.DataFrame(y1))
    y_train = sc_y.inverse_transform(pd.DataFrame(y))

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('ceemdan_eemd_lstm_results.csv', index=False)

    mape = np.mean((np.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def proposed_method(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    from PyEMD import EMD, EEMD, CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf = pd.DataFrame(IMFs)
    ceemdan1 = full_imf.T

    imf1 = ceemdan1.iloc[:, 0]
    imf_dataps = numpy.array(imf1)
    imf_datasetss = imf_dataps.reshape(-1, 1)
    imf_new_datasets = pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1, skipna=True)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 64
    lr = 0.001
    optimizer = 'Adam'

    for col in new_ceemdan:
        datasetss2 = pd.DataFrame(new_ceemdan[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
        from tensorflow.python.keras.layers.recurrent import LSTM

        neuron = 128
        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('Proposed_results.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)

def proposed_method1(new_data, i, look_back, data_partition, cap):
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()

    datas = data1['co2']
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    from PyEMD import EMD, EEMD, CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf = pd.DataFrame(IMFs)
    ceemdan1 = full_imf.T

    imf1 = ceemdan1.iloc[:, 0]
    imf_dataps = numpy.array(imf1)
    imf_datasetss = imf_dataps.reshape(-1, 1)
    imf_new_datasets = pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1, skipna=True)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    epoch = 100
    batch_size = 128
    lr = 0.001
    optimizer = 'Adam'

    for col in new_ceemdan:
        datasetss2 = pd.DataFrame(new_ceemdan[col])
        datasets = datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # BiLSTM
        from keras.models import Sequential
        from tensorflow.keras.layers import Dense, Bidirectional, LSTM
        from sklearn import metrics
        from sklearn.metrics import mean_squared_error, r2_score
        from math import sqrt

        neuron = 128
        model = Sequential()
        model.add(Bidirectional(LSTM(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2]))))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # 将 y_test 和 a 保存到 CSV 文件中
    result_df = pd.concat([y_test, a], axis=1)
    result_df.columns = ['y_test', 'a']  # 根据需要为列命名
    result_df.to_csv('ceemdan_ewt_bitcn.csv', index=False)

    # summarize the fit of the model
    mape = numpy.mean((numpy.abs(y_test - a)) / y_test) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)
    r2 = r2_score(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R-squared score:', r2)


