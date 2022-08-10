
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

with open(('MarvellousHeadBrainDataset.csv'),'r') as file:
    data = pd.read_csv(file)
    df=pd.DataFrame(data)
    print(df.head())#Display First Five elemenst
    
    X=df['Head Size(cm^3)'].values
    Y=df['Brain Weight(grams)'].values
    
    X=X.reshape((-1,1))
    
    reg=LinearRegression()
    reg.fit(X,Y)
    Y_prediction=reg.predict(X)
    #print("Y_prediction",Y_prediction)

    r2=reg.score(X,Y)
    print("r2",r2)
    