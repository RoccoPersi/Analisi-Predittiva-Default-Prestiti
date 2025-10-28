import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loan_default = pd.read_csv("loan_default_final.csv")

#Importo i vari modelli per confrontarli

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

X = loan_default.drop('Status',axis=1)
y = loan_default['Status']

RS = 123 #imposto un seme in modo che le divisioni del train e test set non cambino ad ogni esecuzione 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=RS)

classifiers = [
	DecisionTreeClassifier(),
	RandomForestClassifier(),
	AdaBoostClassifier(),
	GradientBoostingClassifier(),
	LogisticRegression()
]

#Addestro ogni classificatore e per ognuno calcolo l'accuracy e il log loss
#Confronto i vari modelli :

for clf in classifiers:
	clf.fit(X_train,y_train)
	name = clf.__class__.__name__

	print("="*30)
	print(name)

	print("*****Results****")
	y_pred = clf.predict(X_test)
	acc = accuracy_score(y_test,y_pred)
	print("Accuracy: "+str(acc))

	y_pred = clf.predict_proba(X_test)
	ll = log_loss(y_test,y_pred)
	print("Log Loss: "+str(ll))

	print("="*30)

	#I punteggi migliori per l'accuratezza saranno quelli più vicini a 1
	#I punteggi migliori per la log loss saranno quelli più bassi


