import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import pandas as pd
from tkinter import filedialog


class LinearRegression:
    def __init__(self):
        self.data = None
        self.X = None
        self.Y = None
        self.weights = None
        self.error = None
        self.read_data()

    def read_data(self):
        # reads excel files
        data = pd.read_csv("Class demo 1, car dealership.csv")

        # Drop the missing values
        data = data.dropna()

        self.data = data
        # print(data)

    def clean_data(self, normalize=False):
        self.X = np.array([self.data.iloc[:, 0].to_numpy()]).T
        self.Y = np.array([self.data.iloc[:, 1].to_numpy()]).T
        bias = np.full((1, len(self.X)), 1)  # creating bias vector based on row length
        if normalize is False:
            self.X = np.insert(self.X, 0, bias, axis=1)  # putting bias vector as first column in X

        else:
            self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
            self.Y = (self.Y - self.Y.mean(axis=0)) / self.Y.std(axis=0)
            self.X = np.insert(self.X, 0, bias, axis=1)  # putting bias vector as first column in X

    def polynomial_regression(self):
        pass

    def scatter_plot(self, x, y, c="red", nameX="x-data", nameY="y-data", marker='o'):
        plt.xlabel(x.columns[0])
        plt.ylabel(y.columns[0])
        s = len(x)
        plt.scatter(x, y, s, c, marker)
        plt.show()

    def gradient_descent(self, alpha, iters):
        X = self.X
        w = np.zeros(len(self.X[0]))
        print(w)
        w = np.array([w]) # making weight vector 2-D
        m = len(X)  # number of data inputs
        temp = 0
        for i in range(iters):
            w = w - (alpha / m) * np.matmul(X.T, np.matmul(X, w.T) - self.Y).T
            J = self.cost_function(w)
            temp = J
            print("MSE = {}".format(J))
        w=w.flatten()
        print("Y = {} + {}x".format(w[0], w[1]))
        return w

    def cost_function(self, theta):
        m = len(self.X)
        e = np.matmul(self.X, theta.T) - self.Y  # error terms
        J = (1 / (2 * m)) * np.sum(np.square(e), axis=0)[0]
        return J


obj = LinearRegression()
obj.clean_data(True)
# obj.scatter_plot(obj.X, obj.Y)
obj.gradient_descent(0.01, 1000)
