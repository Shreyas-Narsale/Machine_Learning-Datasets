

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
def DecisionTreeAccuracy():
    iris=load_iris()
    
    target=iris.target
    Data=iris.data
    
    data_train, data_test,target_train, target_test=train_test_split(Data,target,test_size=0.5)
    
    classifier=tree.DecisionTreeClassifier()
    
    classifier.fit(data_train,target_train)
    
    predict=classifier.predict(data_test)
    
    Accuracy=accuracy_score(target_test,predict)
    
    return Accuracy
    
def KNeighborsClassifierAccuracy():
    iris=load_iris()
    
    target=iris.target
    Data=iris.data
    
    data_train, data_test,target_train, target_test=train_test_split(Data,target,test_size=0.5)
    
    classifier=KNeighborsClassifier()
    
    classifier.fit(data_train,target_train)
    
    predict=classifier.predict(data_test)
    
    Accuracy=accuracy_score(target_test,predict)
    
    return Accuracy
def main():
    Accuracy=DecisionTreeAccuracy()
    print("DecisionTreeAccuracy:",Accuracy*100,"%d")
    
    Accuracy=KNeighborsClassifierAccuracy()
    print("KNeighborsClassifierAccuracy:",Accuracy*100,"%d")
    
    
if __name__=="__main__":
    main()