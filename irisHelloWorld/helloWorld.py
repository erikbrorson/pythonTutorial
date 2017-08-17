
import numpy as np
import pandas as pd
from sklearn import tree

iris = pd.read_csv("/Users/Erik/Dropbox/Python Tutorial/irisHelloWorld/iris.csv")


# Create decision tree 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(y= iris.species, 
                X = iris[["sepalLength", "sepalWidth",
                "petalLength",
                "petalWidth"]])

prediciton = clf.predict(X = iris[["sepalLength", "sepalWidth",
                "petalLength",
                "petalWidth"]])

def numberCorrect(array):
    n = 0
    i = 0
    for x in array:
        if x == iris.species[i]:
            n = n + 1
        i = i + 1
    return n
    

print(numberCorrect(prediciton))


        
    