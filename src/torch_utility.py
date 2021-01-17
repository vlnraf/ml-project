import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def plot_loss(history, val=True):
    plt.title('Loss plot')
    plt.plot(history['loss'])
    if(val):
        plt.plot(history['val_loss'], '--')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train','val'], loc='upper right')

def plot_accuracy(history, val=True):
    plt.title('Accuracy plot')
    plt.plot(history['acc'])
    if(val):
        plt.plot(history['val_acc'], '--')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train','val'], loc='lower right')

def load_monk(dataset="1"):
    X, y = fetch_openml('monks-problems-'+dataset, return_X_y=True)
    y = y.astype(np.float32)

    if(dataset =="1"):
        Xtrain = OneHotEncoder(sparse=False).fit_transform(X[:124])
        Xtest = OneHotEncoder(sparse=False).fit_transform(X[124:])

        ytrain, ytest = y[:124], y[124:]
        ytrain = ytrain.values
        ytest = ytest.values
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        ytest = ytest.reshape(ytest.shape[0], 1)

    elif(dataset =="2"):
        Xtrain = OneHotEncoder(sparse=False).fit_transform(X[:169])
        Xtest = OneHotEncoder(sparse=False).fit_transform(X[169:])

        ytrain, ytest = y[:169], y[169:]
        ytrain = ytrain.values
        ytest = ytest.values
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        ytest = ytest.reshape(ytest.shape[0], 1)

    elif(dataset =="3"):
        Xtrain = OneHotEncoder(sparse=False).fit_transform(X[:122])
        Xtest = OneHotEncoder(sparse=False).fit_transform(X[122:])

        ytrain, ytest = y[:122], y[122:]
        ytrain = ytrain.values
        ytest = ytest.values
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        ytest = ytest.reshape(ytest.shape[0], 1)


    return Xtrain, Xtest, ytrain, ytest

def table_info(data):
    print('\t    MSE     Accuracy')
    print('-----------------------------')
    print('Train\t|%.7f|\t%.2f|'%(data[0], data[1]))
    print('Test\t|%.7f|\t%.2f|'%(data[2], data[3]))
