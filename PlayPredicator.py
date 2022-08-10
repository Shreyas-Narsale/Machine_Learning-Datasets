



import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

with open(('MarvellousInfosystems_PlayPredictor.CSV'),'r') as file:
    data = pd.read_csv(file)
    df=pd.DataFrame(data)
    print(df.head())#Display First Five elemenst
    
    le = preprocessing.LabelEncoder()
    whether = le.fit_transform(df['Wether'])

    Temp= le.fit_transform(df['Temperature'])
    
    target = le.fit_transform(df['Play'])
    
    Data=list(zip(whether,Temp))
    
    data_train, data_test,target_train, target_test=train_test_split(Data,target,test_size=0.5)
    
    classifier=KNeighborsClassifier(n_neighbors=3)
    
    classifier.fit(data_train,target_train)
    
    predict=classifier.predict(data_test)
    
    Accuracy=accuracy_score(target_test,predict)#For Accuracy
    print("Accuracy is:",Accuracy*100)