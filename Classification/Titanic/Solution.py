import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
train_path = "train.csv"
test_path = "test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
#convert values into numeric Form
train["Sex"][train["Sex"] == 'male'] = 0
train["Sex"][train["Sex"] == 'female'] = 1
train["Embarked"] = train["Embarked"].fillna('S')
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train["Age"] = train["Age"].fillna(train["Age"].median())

target = train["Survived"].values

"""feature_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(feature_one, target)"""
#print(my_tree_one.feature_importances_)
#print(my_tree_one.score(feature_one, target))
   
test["Sex"][test["Sex"] == 'male'] = 0
test["Sex"][test["Sex"] == 'female'] = 1
test["Embarked"] = test["Embarked"].fillna('S')
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(train["Age"].median())
#test_feature = test[["Pclass", "Sex", "Age", "Fare"]]
"""my_prediction = my_tree_one.predict(test_feature)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
#print(my_solution)
#print(my_solution.shape)
my_solution.to_csv("result.csv", index_label = ["PassengerId"])"""

"""feature_two = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 50, min_samples_split = 1, random_state = 1)
my_tree_two = my_tree_two.fit(feature_two, target)

print(my_tree_two.score(feature_two, target))"""
"""train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
feature_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "family_size"]]
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(feature_three, target)
print(my_tree_three.score(feature_three, target))"""

train["family_size"] = train["SibSp"] + train["Parch"] + 1
features_forest = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "family_size"]]
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

print(my_forest.score(features_forest, target))
test["family_size"] = test["SibSp"] + test["Parch"] + 1
features_forest_test = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "family_size"]]
solution = my_forest.predict(features_forest_test)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(solution, PassengerId, columns = ["Survived"])
my_solution.to_csv("result.csv", index_label = ["PassengerId"])
