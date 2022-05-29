import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pickle


def Reader(path, node):
    result = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        matrix = np.loadtxt(fpath, usecols=range(28))
        result.append(matrix[node[0]][node[1]])
        # print(matrix)
        # print(matrix.shape)
        # with open(fpath) as f:
        #     lines = f.readlines()
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


if __name__ == "__main__":
    # nodes = [(5, 8), (8, 5), (5, 12), (8, 12)]
    nodes = [(4, 7), (7, 4), (4, 11), (7, 11)]
    n_steps = 3
    accident = 12000
    # vectors = np.array([Reader('Generator ruchu/traffic', n) for n in nodes])
    # pickle.dump(vectors, open('vectors.pkl', 'wb'))

    vectors = pickle.load(open('vectors.pkl', 'rb'))
    print(vectors.shape)
    print(vectors)

    X, y = split_sequence(vectors[0], n_steps)
    print(X)
    print(X[:accident])

    print(X.shape)
    print(X[:accident].shape)
    X_train, X_test, y_train, y_test = X[:accident], X[accident:], y[:accident], y[accident:]

    model = MLPRegressor(random_state=1, max_iter=1000)
    # model.fit(X_train, y_train)

    # predictions = model.predict(X_test)
    # print(y_test)
    # print(predictions)
    # plt.plot(predictions)
    # plt.plot(y_test)
    # plt.show()
    # fig = px.line(predictions)
    # fig.add_trace(y_test)
    # fig.show()
    # TODO ocena predykcji przy pomocy metryki MAPE
    # print(MAPE(y_test, predictions))

    fig = px.line(vectors)
    fig.show()
    # plt.show()
