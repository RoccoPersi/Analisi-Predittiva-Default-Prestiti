import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
from sklearn.ensemble import GradientBoostingClassifier

loan_default = pd.read_csv("loan_default_final.csv")

X = loan_default.drop('Status',axis=1)
y = loan_default['Status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) #Applico Feature scaling sulle feature
X_test_scaled = scaler.transform(X_test)

from sklearn.model_selection import RandomizedSearchCV

gbc = GradientBoostingClassifier()
gbc_params = {"n_estimators":[5,10,100],"max_depth":[1,3,5,7,9],"learning_rate":[0.01,0.1,1,10,100]}

cv = RandomizedSearchCV(gbc,gbc_params,cv=5,scoring='accuracy') #utilizzo RandomSearch per la ricerca degli iperparametri
cv.fit(X_train_scaled,y_train)

print("*"*30)
print("Best Parameters: "+str(cv.best_params_)) 
print("*"*30)	

clf = GradientBoostingClassifier(n_estimators=cv.best_params_['n_estimators'],max_depth=cv.best_params_['max_depth'],learning_rate=cv.best_params_['learning_rate'])
#assegno al classificatore i parametri trovati al passo precedente

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import precision_score,recall_score,f1_score

#Valuto il mio modello in base al valore delle seguenti metriche

acc = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print("="*30)
print("Accuray del modello finale: "+str(acc))
print("Precision del modello finale: "+str(precision))
print("Recall del modello finale: "+str(recall))
print("F1 Score del modello finale: "+str(f1))
print("="*30)
