# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 19:40:24 2022

@author: morais
"""

import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import graphviz


#==============================================================================
#   Carregar a base de dados
#==============================================================================
df=pd.read_csv('Qualidade_Fruta.csv')

#Estatística básica
#estatistica=df.describe()


#==============================================================================
#   Separação dos Atributos previsores e Atributos Meta ou Classe
#==============================================================================
x_df=df.iloc[:,0:4].values
y_df=df.iloc[:,4].values

#==============================================================================
#Transformação dos atributos categóricos
#==============================================================================
from sklearn.preprocessing import LabelEncoder
leCasca=LabelEncoder()
leCor=LabelEncoder()
leTamanho=LabelEncoder()
lePolpa=LabelEncoder()
leRisco=LabelEncoder()

x_df[:,0]=leCasca.fit_transform(x_df[:,0])
x_df[:,1]=leCor.fit_transform(x_df[:,1])
x_df[:,2]=leTamanho.fit_transform(x_df[:,2])
x_df[:,3]=lePolpa.fit_transform(x_df[:,3])

#==============================================================================
#Faz a padronização dos dados
#==============================================================================
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_df=scaler.fit_transform(x_df)

#==============================================================================
#Divisão da base de dados entre treinamento e teste
#==============================================================================
from sklearn.model_selection import  train_test_split
x_treinamento,x_teste,y_treinamento,y_teste=train_test_split(x_df,y_df,test_size=0.20,random_state=0)

#==============================================================================
#   Execução do algorítimo Árvore de decisão
#==============================================================================
from sklearn.tree import DecisionTreeClassifier
arvore=DecisionTreeClassifier(criterion='entropy',random_state=0)
arvore.fit(x_treinamento,y_treinamento)
previsoes=arvore.predict(x_teste)

#==============================================================================
#   Estatística do modelo
#==============================================================================
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
exatidao=accuracy_score(y_teste,previsoes)
cm=confusion_matrix(y_teste,previsoes)
print(classification_report(y_teste,previsoes))


