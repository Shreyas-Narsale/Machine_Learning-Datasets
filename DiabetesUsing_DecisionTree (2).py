import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def DisplayGraphically(neighbors_setting,training_accuracy,testing_accuracy):
    plt.plot(neighbors_setting,training_accuracy,label="training accuracy")
    plt.plot(neighbors_setting,testing_accuracy,label="testing accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.savefig('knn_comapre_model')
    plt.show()
    
def DiabetesKNN():
    diabetes_Data=pd.read_csv("diabetes.csv")
    
    print("Columns of diabetesDataSet")
    print(diabetes_Data.columns)
    
    print("First 5 Records are:",diabetes_Data.head())
    print("Dimensions of Dataset are:",diabetes_Data.shape)
    
    features=diabetes_Data.drop("Outcome",axis=1)
    target=diabetes_Data["Outcome"]
    
    xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.5)
    
    training_accuracy=[]
    testing_accuracy=[]
    #try n_neighbors from 1 to 10
    neighbors_setting=range(1,11)
    
    for n_neighbors in neighbors_setting:
        #Build the model
        knn=KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(xtrain,ytrain)
        #record training set accuracy
        training_accuracy.append(knn.score(xtrain,ytrain))
        #record test set accuracy
        testing_accuracy.append(knn.score(xtest,ytest))
        
    DisplayGraphically(neighbors_setting,training_accuracy,testing_accuracy)
    
    #near k=9 it shows more accuracy 
    knn=KNeighborsClassifier()
    knn.fit(xtrain,ytrain)
    
    print("Accuracy for Training Dataset At default n_neighbors:",knn.score(xtrain,ytrain))
    print("Accuracy for Testing Dataset At default n_neighbors:",knn.score(xtest,ytest))
    
        
    knn=KNeighborsClassifier(n_neighbors=9)
    knn.fit(xtrain,ytrain)
    
    print("Accuracy for Training Dataset At (n_neighbors=9) :",knn.score(xtrain,ytrain))
    print("Accuracy for Testing Dataset At (n_neighbors=9) :",knn.score(xtest,ytest))
    
        
    
    
    
    
    
def main():
    DiabetesKNN()


if __name__=="__main__":
    main()