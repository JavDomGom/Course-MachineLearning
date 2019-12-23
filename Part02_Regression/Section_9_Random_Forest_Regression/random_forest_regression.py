# Regresión con bosques aleatorios

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
'''

# Escalado de variables
'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

# Ajustar la regresión de bosques aleatorios con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators=300, random_state=0)
regression.fit(x, y)

# Predicción de nuestros modelos con bosques aleatorios
y_pred = regression.predict([[6.5]])


# Visualización de los resultados del modelo con bosques aleatorios
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regression.predict(x_grid), color = 'blue')
plt.title('Modelo de regresión con bosques aleatorios')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()