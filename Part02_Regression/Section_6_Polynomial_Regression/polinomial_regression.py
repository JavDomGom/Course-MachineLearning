# Regresión polinómica

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

## Instancia objeto de la clase StandardScaler
sc_x = StandardScaler()

## Se escalan los valores de la matriz de datos "x"
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Ajustar la regresión plonómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 7)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualización de los resultados del modelo lineal
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Modelo de regresión lineal')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()

# Visualización de los resultados del modelo polinómico
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Modelo de regresión polinómica')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()

# Predicción de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


