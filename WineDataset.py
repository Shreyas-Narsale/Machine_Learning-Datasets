



import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

with open(('WinePredictor.CSV'),'r') as file:
    data = pd.read_csv(file)
    df=pd.DataFrame(data)
    print(df.head())#Display First Five elemenst
    
    le = preprocessing.LabelEncoder()
    alcohol = le.fit_transform(df['Alcohol'])

    malic_acid= le.fit_transform(df['Malic acid'])
    
    ash= le.fit_transform(df['Ash'])
    
    alcalinity_of_ash=le.fit_transform(df['Alcalinity of ash'])

    magnesium= le.fit_transform(df['Magnesium'])
    
    total_phenols= le.fit_transform(df['Total phenols'])
    
    flavanoids= le.fit_transform(df['Flavanoids'])
    
    nonflavanoid_phenols= le.fit_transform(df['Nonflavanoid phenols'])
    
    proanthocyanins= le.fit_transform(df['Proanthocyanins'])
    
    color_intensity= le.fit_transform(df['Color intensity'])
    
    hue= le.fit_transform(df['Hue'])
    
    diluted_wines= le.fit_transform(df['OD280/OD315 of diluted wines'])
    
    proline= le.fit_transform(df['Proline'])
    
    target=le.fit_transform(df['Class'])
    
    Data=list(zip(alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,diluted_wines,proline))
    
    data_train, data_test,target_train, target_test=train_test_split(Data,target,test_size=0.5)
    
    classifier=KNeighborsClassifier(n_neighbors=5)
    
    classifier.fit(data_train,target_train)
    
    predict=classifier.predict(data_test)
    
    Accuracy=accuracy_score(target_test,predict)#For Accuracy
    print("Accuracy is:",Accuracy*100)