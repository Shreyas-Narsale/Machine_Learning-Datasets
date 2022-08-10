

from sklearn import tree
def MarvellousML(weight,surface):
    BallsFeature=[[35,1],[47,1],[90,0], [48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    
    Labels=[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]
    
    classifier=tree.DecisionTreeClassifier()
    
    model=classifier.fit(BallsFeature,Labels)
    
    prediction=classifier.predict([[weight,surface]])
    
    if (prediction==1):
        print("Tennis Ball")
    else:
        print("Cricket Ball")
        
def main():
    weight=input("Enter Testing Weight:")
    
    SurfaceAra=input("Enter Testing SurfaceAra:")
    
    if(SurfaceAra.lower()=="rough"):
        SurfaceAra=1
    elif(SurfaceAra.lower()=="smooth") :
        SurfaceAra=0
    else:
        print("Enter valid SurfaceAra")
        
    MarvellousML(weight,SurfaceAra)
        
    




if __name__=="__main__":
    main()