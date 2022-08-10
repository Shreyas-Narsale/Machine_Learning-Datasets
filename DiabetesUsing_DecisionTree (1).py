#For Featurre Importance refer this file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys
import numpy as np
import matplotlib.pyplot as plt



def plot_feature_importance(model):
    plt.figure()
    n_features=len(model.feature_importances_)
    plt.barh(range(n_features),model.feature_importances_,align='center')
    features=model.feature_names_in_
    plt.yticks(np.arange(n_features),features)
    plt.xlabel("feature_importances")
    plt.ylabel("feature")
    plt.ylim(-1,n_features)
    plt.show()
    
    
def DiabetesDecisionTree():
    diabetes_Data=pd.read_csv("diabetes.csv")
    
    print("Columns of diabetesDataSet")
    print(diabetes_Data.columns)
    
    print("First 5 Records are:",diabetes_Data.head())
    print("Dimensions of Dataset are:",diabetes_Data.shape)
    
    features=diabetes_Data.drop("Outcome",axis=1)
    target=diabetes_Data["Outcome"]
    
    xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.5)
    
    tree=DecisionTreeClassifier(random_state=0)
    tree.fit(xtrain,ytrain)
    
    print("Accuracy for Training Dataset:",tree.score(xtrain,ytrain))
    print("Accuracy for Testing Dataset:",tree.score(xtest,ytest))
    
    tree=DecisionTreeClassifier(max_depth=3,random_state=0)
    tree.fit(xtrain,ytrain)
    
    print("Accuracy for Training Dataset:",tree.score(xtrain,ytrain))
    print("Accuracy for Testing Dataset:",tree.score(xtest,ytest))
    
    print("Feature Importance :",tree.feature_importances_)#Used to refer which feature is importance to take it in our Dataset
    print("Feature Importance Graphically :")
    plot_feature_importance(tree)
    
    
def main():
    DiabetesDecisionTree()


if __name__=="__main__":
    main()