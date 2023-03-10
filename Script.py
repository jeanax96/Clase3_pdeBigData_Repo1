# -*- coding: utf-8 -*-
"""
Spyder Editor

Autor: Jean Carlos Campos
"""
# %% Cargamos modulos y datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

#Modulos para ML

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Datos

from sklearn.datasets import load_digits

digits = load_digits()

# Agregamos un conjunto de datos para realizar una regresion logistica multiclase

# %% Veamos algo de los datos

dir(digits)

# Descripcion del dataset
print(digits["DESCR"])

# data
digits["data"]
digits["data"].shape

digits["data"][0].shape

# images

digits["images"].shape
digits["images"][0].shape

# target

digits["target"]
digits["target"][0]
digits["target_names"][0]

# visualizemos la primera observacion

ind = 999
plt.imshow(digits["images"][ind], cmap = plt.cm.Greys)
plt.title(digits["target"][ind])

# %% Regresion Logistica Multiclase

# Particionamiento de los datos (digits) en train/test

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25)

# Construccion del modelo: Instaciamos la clase LogisticRegression
# Luego de una primera ejecucion, nos dimos cuenta de que se supera el numero de interaciones
model1_digits = LogisticRegression(max_iter = 3000)

# Ajustamos el modelo a nuestro conjunto de datos de entrenamiento

model1_digits.fit(x_train, y_train)

# Hagamos predicciones con el dataset de testeo

y_pred = model1_digits.predict(x_test)

# %% Validacion del modelo

# Metodo score

score = model1_digits.score(x_test, y_test)
print(score)

# Matriz de confusion
from sklearn import metrics

metrics.confusion_matrix(y_test, y_pred)


