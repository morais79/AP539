# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:44:02 2022

@author: morais
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#==============================================================================
#   CARREGAR A BASE DE DADOS 
#==============================================================================
df=pd.read_csv('soybean_bruto.csv')

#==============================================================================
#   DETERMINAR OS VALORES FALTANTES 
#==============================================================================
df=df.replace({"?":np.nan})
valoresFaltantes=df.isna().sum()

#converter para o formato adequado
vf=valoresFaltantes.values

#percentual de valores faltantes
percentualValoresFaltantes=(vf/len(df))*100

#baseDados.fillna(baseDados.mode(),inplace=True)
#baseDados['plant-stand'].fillna(baseDados['plant-stand'].mode(),inplace=True)
teste1=df['plant-stand']
teste3=df['plant-stand'].value_counts()

moda=df['plant-stand'].mode()
df['plant-stand']=df['plant-stand'].fillna(moda)
#df['plant-stand'].fillna(df['plant-stand'].mode(),inplace=True)




#Estatística básica
#estatistica=baseDados.describe()

#correlacao=baseDados.corr()
#testeCor=baseDados['D1_0_gap'].corr(baseDados['sNumber'])