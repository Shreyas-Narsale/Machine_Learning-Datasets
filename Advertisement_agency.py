
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

with open(('Advertising.csv'),'r') as file:
    data = pd.read_csv(file)
    df=pd.DataFrame(data)
    print(df.head())#Display First Five elemenst
    
    Data=df['TV']
    target=df['sales']
    
    # Function Labeling
    plt.title('advertisement agency Graph')
    plt.xlabel("Tv")
    plt.ylabel("Sales")
    #Function to Plot
    plt.scatter(Data,target)
    # function to show the plot
    plt.show()
    
    X=df['TV'].values
    Y=df['sales'].values
    
    X=X.reshape((-1,1))
    
    reg=LinearRegression()
    reg.fit(X,Y)
    Y_prediction=reg.predict(X)
    
    r2=reg.score(X,Y)
    print("r2 is:",r2)
    
    

    #print(Data)
    #print(target)