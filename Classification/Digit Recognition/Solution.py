import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np

class Solution:

    def __init__(self):
        self.train = pd.read_csv("train.csv")
        self.VariableCreation()

    def VariableCreation(self):
        self.y = self.train["label"].values
        self.x = self.train.ix[:, 1:]
        #print(self.x.shape)
        #print(self.y.shape)
        #print(self.x.head())
        self.Neutral()

    def Neutral(self):
        """self.mlp = MLPClassifier(hidden_layer_sizes=(50,),activation='logistic', max_iter=1000, alpha=1e-4,
                                 solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.001)
        self.mlp.fit(self.x, self.y)
        print(self.mlp.score(self.x, self.y))"""
        self.sv = svm.SVC()
        self.sv.fit(self.x, self.y)
        self.Predictions()

    def Predictions(self):
        test = pd.read_csv("test.csv")
        #solution = self.mlp.predict(test)
        solution = self.sv.predict(test)
        w = pd.read_csv("sample_submission.csv")
        Indexing = np.array(w["ImageId"]).astype(int)
        my_solution = pd.DataFrame(solution, Indexing, columns=["Label"])
        my_solution.to_csv("result.csv", index_label=["ImageId"])