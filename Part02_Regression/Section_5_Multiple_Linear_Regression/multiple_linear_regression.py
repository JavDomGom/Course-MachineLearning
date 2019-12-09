# Regresión lineal múltiple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


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

## Cambiar datos categóricos de la columna 3 por valores numéricos
x[:, 3] = le_x.fit_transform(x[:, 3])

'''
Instancia objeto de la clase make_column_transformer a la
que se le pasa la columna con la que se va a trabajar "3"
'''
onehotencoder = make_column_transformer((OneHotEncoder(categories='auto'), [3]), remainder = "passthrough")

## Transformación de todo x
x = onehotencoder.fit_transform(x)

'''
Evitar la trampa de las variables ficticias o dummy.
Hay que eliminar una. En este caso la primera columna.
'''
x = x [:, 1:]


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

'''
La función train_test_split devuelve 4 datos, se le ha de indicar en primer
lugar la matriz de datos "x", a continuación se le suministra la matriz de
datos "y", es decir, el vector de datos que queremos predecir, después hay que
indicar el valor test_size que representará el porcentaje de observaciones que
se emplearán para testing, en este caso un 0.2 (20%), el otro 0.8 (80%) se
empleará como conjunto de entrenamiento. Finalmente se indica una variable
random_state, se trata de un número para poder reproducir el algoritmo. Esta
propiedad se utiliza para emplear una semilla y que en cada ejecución devuelva
siempre los mismos datos, de lo contrario, si no especificamos este atributo
obtendrá un valor aleatorio yen cada ejecución devolverá datos diferentes. En
este ejemplo se ha utilizado el número 0, pero se puede poner cualquiera.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# =============================================================================
# # Escalado de variables
# from sklearn.preprocessing import StandardScaler
# 
# ## Instancia objeto de la clase StandardScaler
# sc_x = StandardScaler()
# 
# ## Se escalan los valores de la matriz de datos "x"
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.transform(x_test)
# =============================================================================

# Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

