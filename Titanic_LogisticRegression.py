import pandas as pd
from matplotlib.pyplot import figure, show
from seaborn import countplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def CleanData(titanic_data):
    titanic_data.drop("PassengerId",axis=1,inplace=True)#Drop the First(zeroth) Column
    #Beacause First columns is of passenger id which is not important
    
    print("First 5 entries after removeing zero columns")
    print(titanic_data.head())
    
    print("Values of Sex Column")
    print(titanic_data["Sex"])
    
    print("Values of Sex Column after removeing one field")
    sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)#Convert categorical variable into dummy/indicator variables.
    #Covert to Data into catorgical columns and then remove one columns
    print(sex.head())
    
    print("Values of Pclass Column")
    print(titanic_data["Pclass"])
    
    print("Values of Pclass Column after removeing one field")
    pclass=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(pclass.head())
    
    print("Values of data set after concatenating new columns")
    titanic_data=pd.concat([titanic_data,sex,pclass],axis=1)
    print(titanic_data.head())
    
    print("Values of data set after removeing one field")
    titanic_data.drop(["Sex","SibSp","Parch","Embarked","Name","Cabin","Ticket"],axis=1,inplace=True)
    print(titanic_data.head())
    
    return titanic_data

def DisplayData(titanic_data):
    print("Visulization :Survived and Non Survived Passenegers")
    figure()#used  to create new figure
    target="Survived"
    countplot(data=titanic_data,x=target).set_title("Survived and Non Survived Passenegers")
    show()#to show figure
    
    print("Visulization :Survived and Non Survived Passenegers on gender")
    figure()
    target="Survived"
    countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and Non Survived Passenegers on gender")
    show()#to show figure
    
    print("Visulization :Survived and Non Survived Passenegers on Passenegers Class")
    figure()
    target="Survived"
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and Non Survived Passenegers on Passenegers Class")
    show()#to show figure
    
    print("Visulization :Survived and Non Survived Passenegers on Passenegers on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and Non Survived Passenegers on Age")# Draw histogram of the DataFrame’s columns using pandas
    show()#to show figure
    
    print("Visulization :Survived and Non Survived Passenegers on Passenegers on Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and Non Survived Passenegers on Fare")
    show()#to show figure

def MarvellousTitanicLogistic():
    #1.Read Csv
    titanic_data=pd.read_csv("train.csv")
    print("First Five Elemenst of Titanic:")
    print(titanic_data.head())
    
    print("No. of Passeneges:",len(titanic_data))
    
    #2.Analyze Data
    DisplayData(titanic_data)
    
    #3.Data Cleaning
    titanic_data=CleanData(titanic_data)
    
    #4.Data Training
    
    x=titanic_data.drop("Survived",axis=1)
    y=titanic_data["Survived"]
    
    # Age Data contain 0 or Nan in it so replace 0 and Nan with mean
    mean_Age = x["Age"].mean(skipna=True)#calcalute Mean
    x["Age"]=x["Age"].replace(0,mean_Age)#replace 0 with mean
    x["Age"]=x["Age"].fillna(mean_Age) #replace Nan with mean
    
    
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)
    
    logmodel=LogisticRegression()
    
    logmodel.fit(xtrain,ytrain)
    
    #4. Data Testing
    predication=logmodel.predict(xtest)
    
    
    #5.Calculate Accruacy
    print("Classification Report of Logistic Regression:")
    print(classification_report(ytest,predication))
    
    print("Confusion Matrix of Logistic Regression:")
    print(confusion_matrix(ytest,predication))
    
    print("Accuracy Score of Logistic Regression:")
    print(accuracy_score(ytest,predication))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def main():
    MarvellousTitanicLogistic()
    
if __name__=="__main__":
    main()