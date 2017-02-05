import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

class Solution:

    def __init__(self):
        self.data = pd.read_csv("DataSets/ex2data1.txt")
        self.data = np.asmatrix(self.data)
        #print(self.data.shape)
        #print(type(self.data))
        #print(self.data[:10])
        self.varableCreation()

    def varableCreation(self):
        [self.m, self.n] = self.data.shape
        self.theta = np.random.randn(1, self.n)
        #print(self.theta.shape)
        self.x = self.data[:, :2]
        self.y = self.data[:, 2]
        self.x = np.c_[np.ones(self.m), self.x]
        self.Gradient()
        #print(self.x.shape)
        #print(self.y.shape)
        #print(self.x[:10])

    def Hypothesis(self, z):
        return 1 / (1 + (np.exp(-z)))

    def Gradient(self):
        alpha = 0.01
        #print(self.m)
        for i in range(1500):
            #print(i)
            a = np.transpose(self.theta)
            b = np.dot(self.x , a)
            h = self.Hypothesis(b)
            c = h - self.y
            d = np.transpose(c)
            e = np.dot(d, self.x)
            f = alpha / self.m
            g = f * e
            self.theta = self.theta - g
            #print(self.theta)

    def Predict(self, test):
        [p, q] = test.shape
        test = np.c_(np.ones(p), test)
        w = np.transpose(self.theta)
        r = np.dot(test, w)
        return self.Hypothesis(r)