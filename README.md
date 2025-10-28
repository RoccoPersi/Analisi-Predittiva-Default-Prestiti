# Analisi Predittiva del Rischio di Default sui Prestiti (Progetto Big Data & BI)

Questo repository contiene il progetto svolto per il corso di Big Data & Business Intelligence, focalizzato sulla predizione del rischio di default per i prestiti utilizzando tecniche di machine learning. Il progetto segue una pipeline completa, dalla preparazione dei dati alla valutazione del modello finale.

# Obiettivo

L'obiettivo principale è costruire un modello predittivo in grado di classificare se un richiedente sarà inadempiente (Status = 1) o meno (Status = 0) basandosi sulle caratteristiche del prestito e del richiedente presenti nel dataset Loan_Default.csv.

# Pipeline del Progetto

Il progetto è strutturato nei seguenti script Python, da eseguire in sequenza:

1_data_preprocessing.py:

Caricamento ed Esplorazione: Carica il dataset Loan_Default.csv e ne esplora le caratteristiche iniziali (tipi di dati, valori mancanti). [Immagine del processo di esplorazione dati]

Pulizia dei Dati: Gestisce i valori mancanti (rimozione per basse percentuali, imputazione con mediana per numerici e moda per categorici).

Gestione Outlier: Identifica e rimuove i valori anomali nelle feature numeriche (utilizzando il criterio delle 3 deviazioni standard dalla media).

Gestione Duplicati: Rimuove eventuali righe duplicate (escluso l'ID).

Output: Salva il dataset pulito in loan_default_clean.csv.

2_ohe_feature.py:

One-Hot Encoding: Applica la codifica One-Hot alle feature categoriche del dataset pulito per trasformarle in formato numerico. [Immagine della codifica One-Hot Encoding]

Feature Selection (Mutual Information): Calcola i punteggi di Mutual Information tra le feature e la variabile target (Status) per valutare la rilevanza di ciascuna feature. Esegue un confronto preliminare (utilizzando Decision Tree) su dataset con diverse selezioni di feature, optando per l'utilizzo del dataset completo.

Output: Salva il dataset finale, pronto per la modellazione, in loan_default_final.csv.

3_model_comparison.py:

Divisione Dati: Suddivide il dataset loan_default_final.csv in set di training e test.

Confronto Modelli: Addestra e valuta diversi modelli di classificazione di base:

Decision Tree Classifier

Random Forest Classifier

AdaBoost Classifier

Gradient Boosting Classifier

Logistic Regression

Valutazione Preliminare: Confronta i modelli utilizzando le metriche di Accuracy e Log Loss sul set di test per identificare il candidato migliore (Gradient Boosting Classifier risulta il più performante). [Immagine delle metriche di confronto modelli]

4_final_model.py:

Feature Scaling: Applica StandardScaler alle feature numeriche dei set di training e test per standardizzarle.

Hyperparameter Tuning: Utilizza RandomizedSearchCV per trovare la combinazione ottimale di iperparametri (n_estimators, max_depth, learning_rate) per il GradientBoostingClassifier, basandosi sull'accuracy e la validazione incrociata (cv=5).

Addestramento Modello Finale: Addestra il GradientBoostingClassifier con i migliori iperparametri trovati sul set di training completo (non scalato, come nel codice fornito - nota: potrebbe essere utile addestrare sui dati scalati se RandomizedSearch è stato fatto su quelli).

Valutazione Finale: Valuta le performance del modello finale sul set di test utilizzando metriche di classificazione complete: Accuracy, Precision, Recall e F1-Score. Stampa i risultati finali.

LoanDefault_RoccoPersiani.pdf: Report dettagliato che descrive tutte le fasi del progetto, le motivazioni delle scelte e l'analisi dei risultati.

Tecnologie Utilizzate

Python 3

# Librerie Principali:

Pandas (per la manipolazione dei dati)

NumPy (per calcoli numerici)

Matplotlib (per la visualizzazione, es. in preprocessing)

Scikit-learn (per preprocessing, feature selection, modelli, tuning e valutazione)

Utilizzo

Prerequisiti:

Avere Python 3 installato.

Installare le librerie necessarie

Avere il dataset originale Loan_Default.csv nella stessa cartella degli script.

# Esecuzione:
Eseguire gli script Python nell'ordine numerico dalla riga di comando:

python 1_data_preprocessing.py
python 2_ohe_feature.py
python 3_model_comparison.py
python 4_final_model.py


Ogni script genera file CSV intermedi (loan_default_clean.csv, loan_default_final.csv) necessari per lo script successivo. L'ultimo script (4_final_model.py) stamperà a console i migliori iperparametri trovati e le metriche di valutazione finali del modello Gradient Boosting ottimizzato.

Per un'analisi approfondita delle scelte e dei risultati, consultare il file LoanDefault_RoccoPersiani.pdf.
