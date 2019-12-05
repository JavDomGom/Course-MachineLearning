# Plantilla de preprocesado - Datos categóricos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Codificar datos categóricos
'''
Para utilizar one hot encoder y crear variables dummy, para
aplicar la dummyficación a la primera columna y dejar el resto
de columnas como están, lo podemos hacer con:
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

## Instancia objeto de la clase LabelEncoder
le_x = LabelEncoder()

## Cambiar datos categóricos de la columna 0 por valores numéricos
x[:, 0] = le_x.fit_transform(x[:, 0])

'''
Instancia objeto de la clase make_column_transformer a la
que se le pasa la columna con la que se va a trabajar "0"
'''
onehotencoder = make_column_transformer((OneHotEncoder(categories='auto'), [0]), remainder = "passthrough")

## Transformación de todo x
x = onehotencoder.fit_transform(x)

'''
Ahora hacemos lo mismo con los datos de y, es
decir, para la columna "purchased" del dataset.
'''
## Instancia objeto de la clase LabelEncoder
le_y = LabelEncoder()

## Cambiar datos categóricos de todo y
y = le_y.fit_transform(y)