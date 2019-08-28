import numpy as np
import pandas as pd
#Inputting and selection of data
DAT = input("Input the name of the file you want predict: ")
type_ = input("What'sthe format of the file: \'a\' for CSV, \'b\' for excel: ")
if type_ == 'a':
    DAT = 'C:/Users/Ademola/Desktop/dataset/'+DAT+'.csv'
    data = pd.read_csv(DAT)
if type_ == 'b':
    DAT = 'C:/Users/Ademola/Desktop/dataset/'+DAT+'.xlsx'
    data = pd.read_excel(DAT,sheetname=None)
else:
    print('restart and input a file name please')
print(data.head())
#Choosing the columns to predict
n = int(input('enter the number of columns you want to predict: '))
a = []
for i in range(n):
    set_name = input('write the name of the data that you wanna predict: ')
    a.append(set_name)
yy = data[a]
X = data.drop(a,axis=1)
c=[]
for i in range(len(X.columns)):
    if (type(X.iloc[2,i])== np.int64 or type(X.iloc[2,i])== np.float64):
     c.append(X.columns[i])
X = data[c]

print('The X values are :','\n', X.head())
ans = input("Is it satisfactory? Type 'no' or 'n' if not satisfied, type any other letter if satisfied: ")
if ans == 'n' or ans == 'no':
   an= int(input('how many columns do you want? '))
   for i in range(an):
       ans = input ('choose your columns ')
       c=[]
       c.append(ans)
   X = data[c]
   
#Data preprocessing
from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
sca.fit(X)
sca_features= sca.transform(X)
df_feat = pd.DataFrame(sca_features,columns=c)
print(df_feat.head(5))
X = df_feat
jj =[]

for i in range(len(yy.columns)):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    yj = labelencoder.fit_transform(yy.iloc[:,i])
    jj.append(yj)
yaa= pd.DataFrame(jj)
yaa = yaa.T

for i in range(0,len(yaa.columns)):
 y = yaa.iloc[:,i]
 #Data Splitting
 from sklearn.cross_validation import train_test_split
 X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)
    
 def classification(data,prediction,y_test,n,y):
    try:
          if (type(y[2]) == np.int64 or type(y[2]) == np.float64):
            from sklearn.metrics import classification_report
            print('Classification Report: ','\n',classification_report(y_test,prediction))
            print('\n')
            from sklearn.metrics import confusion_matrix
            print('Confusion Matrix is: ','\n', confusion_matrix(y_test,prediction))
            from sklearn.metrics import accuracy_score
            print('Accuracy Score is:', accuracy_score(y_test,prediction))
            from sklearn.metrics import mean_squared_error
            print('Mean Squared Error: ',mean_squared_error(y_test,prediction))
            return mean_squared_error
            return classification_report
            return confusion_matrix
            return accuracy_score
    except ValueError:
          if (type(y[2]) == np.int64 or type(y[2]) == np.float64):
            from sklearn.metrics import mean_squared_error,mean_absolute_error
            print('This solver needs samples of at least 2 different classes in the data,','Mean Absolute Error: ',mean_absolute_error(y_test,prediction))
            print('This solver needs samples of at least 2 different classes in the data,','Mean Squared Error: ',mean_squared_error(y_test,prediction))
            return mean_squared_error
            return mean_absolute_error
          else:
            print("Can't use string data for Mean squared or Mean absolute error")  
    except:
        print('something went wrong, sorry!')
    
 def linearregression(data,X_train,y_train,X_test,y_test,n,y):
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        lm.fit(X_train,y_train)
        kk = input('Type "yes" or "y" if you want the see the linear intercepts:')
        if kk== 'y' or kk=='yes':
            print(lm.intercept_)
        pred1 =lm.predict(X_test)
        y_test = labelencoder.inverse_transform(y_test)
        y = labelencoder.inverse_transform(y)
        ky = input("Type 'yes' or 'y' if you want the see the linear regression predicted values, type any other letter if 'no':")
        if ky== 'y' or ky=='yes':
            print(pred1)
        print('linear_regression: ','\n', classification(data,pred1,y_test,n,y))
      
 def logisticregression(data,X_train,y_train,X_test,y_test,n,y):
        from sklearn.linear_model import LogisticRegression
        logmodel = LogisticRegression()
        logmodel.fit(X_train,y_train)
        pred = logmodel.predict(X_test)
        pred2 = labelencoder.inverse_transform(pred)
        y_test = labelencoder.inverse_transform(y_test)
        y = labelencoder.inverse_transform(y)
        ky = input("Type 'yes' or 'y' if you want the see the logistics regression predicted values, type any other letter if 'no':")
        if ky== 'y' or ky=='yes':
            print(pred2)
        print('logistic_regression: ','\n',classification(data,pred2,y_test,n,y))
        
 def KNN(data,X_train,y_train,X_test,y_test,n,y):
        from sklearn.neighbors import KNeighborsClassifier
        error_rate=[]
        if len(y_test.index) < 500 :
          for i in range(1,len(y_test.index)):
            knn=KNeighborsClassifier(n_neighbors= i)
            knn.fit(X_train,y_train)
            pred_i= knn.predict(X_test)
            error_rate.append(np.mean(pred_i!=y_test))
        else :
          for i in range(1,500):
            knn=KNeighborsClassifier(n_neighbors= i)
            knn.fit(X_train,y_train)
            pred_i= knn.predict(X_test)
            error_rate.append(np.mean(pred_i!=y_test))
        err = pd.DataFrame(error_rate)
        ev2 = err.min(axis=1)
        nm = ev2.idxmin()
        knn = KNeighborsClassifier(n_neighbors=(nm+1))
        knn.fit(X_train,y_train)
        pred = knn.predict(X_test)
        pred3 = labelencoder.inverse_transform(pred)
        y_test = labelencoder.inverse_transform(y_test)
        y = labelencoder.inverse_transform(y)
        ky = input("Type 'yes' or'y' if you want the see the KNN predicted values, type any other letter if 'no':")
        if ky== 'y' or ky=='yes':
            print(pred3)
        print('KNN: ','\n',classification(data,pred3,y_test,n,y))
        
 def trees(data,X_train,y_train,X_test,y_test,n,y):
        from sklearn.tree import DecisionTreeClassifier
        dtree=DecisionTreeClassifier()
        dtree.fit(X_train,y_train)
        pred = dtree.predict(X_test)
        pred4 = labelencoder.inverse_transform(pred)
        y_test = labelencoder.inverse_transform(y_test)
        y = labelencoder.inverse_transform(y)
        ky = input("Type 'yes' or 'y' if you want the see the Tree predicted values, type any other letter if 'no':")
        if ky== 'y' or ky=='yes':
            print(pred4)
        print('Trees: ','\n',classification(data,pred4,y_test,n,y))
        
 def decisiontree(data,X_train,y_train,X_test,y_test,n,y):
        from sklearn.tree import DecisionTreeClassifier
        dtree=DecisionTreeClassifier()
        dtree.fit(X_train,y_train)
        pred = dtree.predict(X_test)
        pred5 = labelencoder.inverse_transform(pred)
        y_test = labelencoder.inverse_transform(y_test)
        y = labelencoder.inverse_transform(y)
        ky = input("Type 'yes' or 'y' if you want the see the Decision tree predicted values, type any other letter if 'no':")
        if ky== 'y' or ky=='yes':
            print(pred5)
        print('Decision_trees: ','\n',classification(data,pred5,y_test,n,y))
    
 def randomforest(data,X_train,y_train,X_test,y_test,n,y):
        from sklearn.ensemble import RandomForestClassifier
        error_rate=[]
        if len(y_test) <500:
          for i in range(1,len(y_test)):
            rfc = RandomForestClassifier(n_estimators=i)
            rfc.fit(X_train,y_train)
            pred_i = rfc.predict(X_test)
            error_rate.append(np.mean(pred_i!=y_test))
        else:
          for i in range(1,500):
            rfc = RandomForestClassifier(n_estimators=i)
            rfc.fit(X_train,y_train)
            pred_i = rfc.predict(X_test)
            error_rate.append(np.mean(pred_i!=y_test))
        nm = error_rate.index(min(error_rate))
        rfc = RandomForestClassifier(n_estimators=(nm+1))
        rfc.fit(X_train,y_train)
        pred = rfc.predict(X_test)
        pred6 = labelencoder.inverse_transform(pred)
        y_test = labelencoder.inverse_transform(y_test)
        y = labelencoder.inverse_transform(y)
        ky = input("Type 'yes' or 'y' if you want the see the Random Forest predicted array, type any other letter if 'no':")
        if ky== 'y' or ky=='yes':
            print(pred6)
        print('Random_forest: ','\n', classification(data,pred6,y_test,n,y))
        
 def man(data,X_train,y_train,X_test,x,y,y_test,n):
    
        linearregression(data,X_train,y_train,X_test,y_test,n,y)
        logisticregression(data,X_train,y_train,X_test,y_test,n,y)
        KNN(data,X_train,y_train,X_test,y_test,n,y)
        trees(data,X_train,y_train,X_test,y_test,n,y)
        decisiontree(data,X_train,y_train,X_test,y_test,n,y)
        randomforest(data,X_train,y_train,X_test,y_test,n,y)
        
 print(man(data,X_train,y_train,X_test,X,y,y_test,n))   
    
    
            
    
        
        
        
        
        
        
        
