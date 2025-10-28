import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Per prima cosa servirà importare il dataset su cui lavorerò attraverso l'apposita funzione messa a disposizione da pandas : read_csv()
loan_default = pd.read_csv("Loan_Default.csv")

print(loan_default.head()) #Con .head() stampo le prime 5 righe che appartengono al dataset
print(loan_default.info()) #.info() mostra da quali colonne è composto il dataset e le loro caratteristiche
print("Numero righe e colonne presenti nel dataframe: "+str(loan_default.shape)) #shape mostra il numero di righe e colonne presenti nel dataframe
print(loan_default.describe())

#Il target del nostro task è la feature Status , quindi il nostro modello dovrà predire la feature Status del dataframe

null_values = loan_default.isnull().sum()
print(null_values) #Per ogni colonna mostra il numero di valori nulli che possiedono le varie colonne che compongono il dataset

cols_list = list() #Inizializzo una lista vuota per memorizzare le colonne con valori mancanti del dataframe
values_list = list() #Inizializzo una lista vuota per memorizzare la % dei valori mancanti per ogni colonna

#Seleziono per ogni colonna la % di valori mancanti che possiede
for col in loan_default.columns:
	pct_missing_values = np.mean(loan_default[col].isnull()*100)
	cols_list.append(col)
	values_list.append(pct_missing_values)

pct_missing_values_df = pd.DataFrame()
pct_missing_values_df['col'] = cols_list
pct_missing_values_df['pct_missing_values'] = values_list #Attraverso queste operazioni creo un nuovo df che contiene la % dei valori mancanti per ogni colonna 

pct_missing_values_df.loc[pct_missing_values_df.pct_missing_values > 0].plot(kind='bar',figsize=(6,6)) #Seleziono e creo un grafico a barre solo delle colonne che hanno una percentuale di missing values superiore a 0
plt.show()

less_pct_mv = list(pct_missing_values_df.loc[(pct_missing_values_df.pct_missing_values < 0.5)&(pct_missing_values_df.pct_missing_values > 0),'col'].values)

loan_default.dropna(subset=less_pct_mv,inplace=True)
#Ho raggruppato tutti i valori che contenevano una % bassa di missing values e in seguito ho eliminato le righe corrispondenti nel dataframe iniziale, iniziando la "pulizia"

#I missing values rimanenti saranno sostituiti :
#Per i valori numerici (quantitativi) ogni valore mancante sarà sostituito con il valore che corrisponde alla mediana della colonna di appartenenza

loan_numeric = loan_default.select_dtypes(include=[np.number])
numeric_cols = loan_numeric.columns.values

for col in numeric_cols:
	missing = loan_default[col].isnull()
	num_missing = np.sum(missing)
	if num_missing > 0: #Sostituisco i missing values con la mediana dei valori della colonna
		med = loan_default[col].median()
		loan_default[col] = loan_default[col].fillna(med)

#Per i valori non numerici (qualitativi) ogni missing values sarà sostituito con la moda della rispettiva colonna

loan_non_numeric = loan_default.select_dtypes(exclude=[np.number])
non_numeric_cols = loan_non_numeric.columns.values

for col in non_numeric_cols:
	missing_v = loan_default[col].isnull()
	num_missing = np.sum(missing)
	if num_missing > 0:
		mod = loan_default[col].describe()['top']
		loan_default[col] = loan_default[col].fillna(mod)

null_val2 = loan_default.isnull().sum()
print("***********VERIFICA***********\n"+str(null_val2)) #Verifico che tutti i valori mancanti siano stati opportunamente trattati

#Visualizziamo attraverso dei grafici i valori per ogni colonna numerica che assumono le rispettive features
for col in numeric_cols:
	loan_default[col].plot(style=".",figsize=(6,6),color="red")
	plt.title(col)
	plt.show()

print(loan_default.shape)

#A questo punto voglio individuare ed eliminare eventuali outliers che potrebbero portarmi overfitting
#Per ogni colonna considererò la sua media e la deviazione standard
#In seguito saranno considerati outliers tutti i valori che si trovano a più di 3 deviazioni standard dalla media
#e verranno rimossi. 

for col in numeric_cols:
	print("Working on column: "+str(col))

	avg = loan_default[col].mean()
	sd = loan_default[col].std()

	loan_default = loan_default[(loan_default[col]<=avg+(3*sd))]

print(loan_default.shape)#Verifico che siano stati effettivamente rimossi

#Come possiamo osservare dai successivi grafici i valori considerati anomali sono stati rimossi con successo

for col in numeric_cols:
	loan_default[col].plot(style=".",figsize=(6,6),color="red")
	plt.title(col)
	plt.show()

#Una volta rimossi i valori anomali andrò a verificare se sono presenti dei valori duplicati all'interno del dataframe
#Anche i duplicati come gli outliers verranno rimossi

cols_other_than_id = list(loan_default.columns)[1:] #Considero tutte le colonne tranne ID poichè è univoco

loan_default.drop_duplicates(subset=cols_other_than_id,inplace=True,keep=False) #Con il metodo .drop_duplicates() elimineremo tutte le righe duplicate se effettivamente sono presenti

loan_default.to_csv('loan_default_clean.csv',encoding='utf-8',index=False) #Salvo le modifiche apportate al dataset creando un nuovo file csv

