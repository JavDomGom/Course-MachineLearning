# Plantilla de regresión

# Regresión polinómica

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('xxxxxxxxx.csv')
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

# Ajustar la regresión con el dataset
# Crear aquí nuestro modelo de regresión
regression = None


# Predicción de nuestros modelos
y_pred = regression.predict([[6.5]])


# Visualización de los resultados del modelo polinómico
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regression.predict(x_grid), color = 'blue')
plt.title('Modelo de regresión')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()
