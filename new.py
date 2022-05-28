import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_absolute_percentage_error
from keras.layers import LSTM, Dense
from keras.models import Sequential
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor


def Reader(path, node):
    result = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        matrix = np.loadtxt(fpath, usecols=range(28))
        result.append(matrix[node[0]][node[1]])
    return result


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
    return mape


def lstm_model(x):
    num_sequences = x.shape[1]
    print(num_sequences)
    num_features = 1

    model = Sequential()
    model.add(
        LSTM(10,
             activation="relu",
             input_shape=(num_sequences, num_features)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def rnn(x):
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # regressor.add(LSTM(units=50, return_sequences=True))
    # regressor.add(Dropout(0.2))
    # regressor.add(LSTM(units=50, return_sequences=True))
    # regressor.add(Dropout(0.2))
    # regressor.add(LSTM(units=50))
    # regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_absolute_percentage_error')

    return regressor


def incremental_fit(regresor, X_to_increment, y_to_increment, batch_len):
    X = [X_to_increment[x:x + batch_len] for x in range(0, len(X_to_increment), batch_len)]
    y = [y_to_increment[x:x + batch_len] for x in range(0, len(y_to_increment), batch_len)]
    plut=[]
    for X_inc, y_inc, idx in zip(X, y, range(len(X))):
        regresor.partial_fit(X_inc, y_inc)
        # y_pred = regresor.predict(X_to_increment.difference(X_inc))

        y_pred = regresor.predict(X_to_increment[(batch_len * idx):])
        # y_pred = regresor.predict(X_to_increment)
        # print(mean_absolute_percentage_error(y_to_increment, y_pred))
        plut.append(mean_absolute_percentage_error(y_to_increment[(batch_len * idx):], y_pred))
        print(idx, ".....", mean_absolute_percentage_error(y_to_increment[(batch_len * idx):], y_pred))
    plt.plot(plut)
    plt.grid()
    plt.show()


def plot_comparison(name, pred, y_test):
    plt.title(name)
    plt.plot(pred)
    plt.plot(y_test)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    nodes = [(5, 8), (8, 5), (5, 12), (8, 12)]
    data_size = 20000
    n_steps = 700

    # steps_vector = [n for n in range(1600, 6000, 50)]

    accident = 11000

    # vectors = np.array([Reader('Generator ruchu/traffic', n) for n in nodes])
    # pickle.dump(vectors, open('vectors.pkl', 'wb'))

    vectors = pickle.load(open('vectors.pkl', 'rb'))
    data = vectors[0]
    # data = np.reshape(data, (-1, 1))

    # print(vectors.shape)
    # print(vectors)

    # plot_acf(vectors[0], lags=11000)
    # plt.show()
    #
    # acf = acf(vectors[0], nlags=2000)
    # print(acf)

    # ploting the traffic
    fig = px.line(vectors.T)

    fig.show()

    # X, y = split_sequence(vectors[0][], n_steps)
    # print("Xshape", X.shape)

    # print(X[:accident])
    ######
    # mape=[]
    # for n_steps in steps_vector:
    X_train, y_train = split_sequence(data[:accident], n_steps)
    X_test, y_test = split_sequence(data[accident:], n_steps)

    ######

    # print(X[:accident].shape)
    # X_train, X_test, y_train, y_test = X[:accident], X[accident:], y[:accident], y[accident:]
    # print("shape", X_train.shape)
    # print("shape", y_train.shape)

    # print("shape", X_test.shape)
    # print("shape", y_test.shape)

    # model = MLPRegressor(random_state=1234, max_iter=1000)
    # model.fit(X_train, y_train)
    #
    # predictions = model.predict(X_test)

    # plt.plot(predictions)
    # plt.plot(y_test)
    # plt.grid()
    # plt.show()
    #
    #     print(n_steps, mean_absolute_percentage_error(y_test, predictions))
    #     mape.append((n_steps,mean_absolute_percentage_error(y_test, predictions)))
    #
    # print(mape)
    # print(sorted(mape, key=lambda l:l[1], reverse=False))

    # lstm_model = lstm_model(X_train)
    # lstm_model.fit(X_train, y_train)
    # lstm_predictions = lstm_model.predict(X_test)
    #
    # print(mean_absolute_percentage_error(y_test, lstm_predictions))
    # plt.plot(lstm_predictions)
    # plt.plot(y_test)
    # plt.grid()
    # plt.show()

    # model2 = rnn(X_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # model2.fit(X_train, y_train, epochs=1)
    # # model2.save('lstm_model.h5')
    # # model2 = load_model('lstm_model.h5')
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # pred2 = model2.predict(X_test)
    # pred2 = np.reshape(pred2, (8300))
    # print(pred2.shape)
    # plt.plot(pred2)
    # plt.plot(y_test)
    # print(y_test.shape)
    # plt.grid()
    # plt.show()

    # regr = RandomForestRegressor(max_depth=10, random_state=1)
    # regr.fit(X_train, y_train)
    # pred = regr.predict(X_test)
    # print(mean_absolute_percentage_error(y_test, pred))
    #
    # plt.plot(pred)
    # plt.title("RFR")
    # plt.plot(y_test)
    # plt.grid()
    # plt.show()

    regr = MLPRegressor(random_state=1234, max_iter=1000)
    # regr = SGDRegressor(random_state=1234, max_iter=1000)
    regr = PassiveAggressiveRegressor(random_state=1234, max_iter=100)
    regr.partial_fit(X_train, y_train)

    pred = regr.predict(X_test)

    # print(mean_absolute_percentage_error(y_test, pred))

    plot_comparison("SGD", pred, y_test)

    # neigh = KNeighborsRegressor(n_neighbors=1000)
    # neigh.fit(X_train, y_train)
    # pred = neigh.predict(X_test)
    # print(mean_absolute_percentage_error(y_test, pred))
    #
    # plt.title("KNN")
    # plt.plot(pred)
    # plt.plot(y_test)
    # plt.grid()
    # plt.show()

    incremental_fit(regr, X_test, y_test, 10)
