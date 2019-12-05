# Plantilla de preprocesado - Datos faltantes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Tratamiento de los NANs
from sklearn.impute import SimpleImputer

## Reemplazar por medias
imputer = SimpleImputer(strategy='mean')

## Medias en todas las filas ":" y solo a columnas 1 y 2 "1:3"
imputer = imputer.fit(x[:, 1:3])

## Cambiar valores por dichas medias
x[:, 1:3] = imputer.transform(x[:, 1:3])