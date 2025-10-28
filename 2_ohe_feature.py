import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

loan_default = pd.read_csv("loan_default_clean.csv")

non_numeric_cols = loan_default.select_dtypes(exclude=np.number)

ohe_encoder = OneHotEncoder()
df = pd.DataFrame(loan_default['ID'])

#Considero solo le colonne non numeriche da codificare
for col in non_numeric_cols: 
	encoder_df = pd.DataFrame(ohe_encoder.fit_transform(loan_default[[col]]).toarray())
	encoder_df.columns = loan_default[col].unique()
	encoder_df.drop_duplicates()
	df[encoder_df.columns] = encoder_df.values

for col in non_numeric_cols:
	loan_default.drop(col,axis=1,inplace=True)

#Unisco le colonne non numeriche con quelle numeriche ottenendo il dataset completo con relative codifiche
final_df = pd.merge(left=loan_default,right=df,on='ID')

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

y = final_df['Status']
X = final_df.drop('Status',axis=1)

mi_scores = mutual_info_classif(X,y) #Calcolo lo score delle features per andare a individuare quali mi danno il risultato migliore 

from sklearn.model_selection import train_test_split

X_train_1,X_test_1,y_train,y_test = train_test_split(X,y,random_state=0,stratify=y)

mi_score_index_selected = np.where(mi_scores > 0.2)[0]
X_2 = X.iloc[:,mi_score_index_selected].values

X_train_2,X_test_2,y_train,y_test = train_test_split(X_2,y,random_state=0,stratify=y)

mi_score_selected_index = np.where(mi_scores < 0.2)[0]

X_3 = X.iloc[:,mi_score_selected_index].values

X_train_3,X_test_3,y_train,y_test = train_test_split(X_3,y,random_state=0,stratify=y)

#Confronto i 3 dataset estratti con un albero decisionale 

from sklearn.tree import DecisionTreeClassifier

model_1 = DecisionTreeClassifier().fit(X_train_1,y_train)
model_2 = DecisionTreeClassifier().fit(X_train_2,y_train)
model_3 = DecisionTreeClassifier().fit(X_train_3,y_train)

score_1 = model_1.score(X_test_1,y_test)
score_2 = model_2.score(X_test_2,y_test)
score_3 = model_3.score(X_test_3,y_test)

print("Score Model 1 : "+str(score_1)+"\n")
print("Score Model 2 : "+str(score_2)+"\n")
print("Score Model 3 : "+str(score_3))

#Come vediamo dai risultati , il modello che ha utilizzato il dataset con miscore < 0.2 ha risultati peggiori rispetto agli altri due 

final_df.to_csv('loan_default_final.csv',encoding='utf-8',index=False)

#Dato che i risultati dei modelli che utilizzano i primi due dataset sono simili opto per utilizzare il dataset completo


	
