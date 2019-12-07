# Regrsión lineal simple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

'''
La función train_test_split devuelve 4 datos, se le ha de indicar en primer
lugar la matriz de datos "x", a continuación se le suministra la matriz de
datos "y", es decir, el vector de datos que queremos predecir, después hay que
indicar el valor test_size que representará el porcentaje de observaciones que
se emplearán para testing, en este caso uno de cada tres (1/3) resto se
empleará como conjunto de entrenamiento. Finalmente se indica una variable
random_state, se trata de un número para poder reproducir el algoritmo. Esta
propiedad se utiliza para emplear una semilla y que en cada ejecución devuelva
siempre los mismos datos, de lo contrario, si no especificamos este atributo
obtendrá un valor aleatorio yen cada ejecución devolverá datos diferentes. En
este ejemplo se ha utilizado el número 0, pero se puede poner cualquiera.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


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

# Crear modelo de regresión lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)