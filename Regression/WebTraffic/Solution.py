import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

#question is approached using Linear Regression


class Solution:
    def __init__(self):
        self.data = sp.genfromtxt("Datasets/web_traffic.tsv", delimiter="\t")
        #print(type(self.data))
        #print(self.data.shape)
        #print(self.data[:10])
        self.RemoveDataAbnorma()
        self.ComputeCalculation()

    def RemoveDataAbnorma(self):
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        #print(self.y[sp.isnan(self.y)])
        #nan count is not great so we can remove rows from data
        self.x = self.x[~sp.isnan(self.y)]
        self.y = self.y[~sp.isnan(self.y)]

    def ComputeCalculation(self):
        self.theta = np.random.randn(2, 1)
        #print(self.theta.shape)
        self.x = np.c_[np.ones(self.x.size), self.x]
        self.GradientDescent()
        #print(self.theta)
        #print(self.x.shape)

    def Hypothesis(self, s):
        return np.dot(s, self.theta)

    def GradientDescent(self):
        alpha = 0.01
        [m, n] = self.x.shape
        para = alpha / m
        for i in range(1500):
            self.theta = self.theta - (para * np.dot(np.transpose(self.x), (self.Hypothesis(self.x) - self.y)))

    def Predict(self, test):
        return self.Hypothesis(test)


