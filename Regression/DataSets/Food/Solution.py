import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd



class Solution:

    def __init__(self):
        self.data = sp.genfromtxt("Dataset.txt", delimiter=",")

    def error(f, x, y):
        return sp.sum((f(x) - y) ** 2)

    def Computaion(self):
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        PlotGraph(self.x, self.y)
        fp = np.polyfit(self.x, self.y, 1, full=False)
        print(fp)
        self.f1 = sp.poly1d(fp)
        # print(error(f1, x, y))

    def predict(self, test):
        return self.f1(test)

def PlotGraph(x, y):
    plt.plot(x, y, 'rx')
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.show()





