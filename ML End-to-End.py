# Databricks notebook source
# MAGIC %md Se actualizan las librerias sólo la primera vez

# COMMAND ----------

#%pip install --upgrade cloudpickle

# COMMAND ----------

# MAGIC %md Se cargan los datos al cluster y luego se leen

# COMMAND ----------

import pandas as pd

red_wine = pd.read_csv("/dbfs/FileStore/datosML/winequality_red.csv", sep = ';')
white_wine = pd.read_csv("/dbfs/FileStore/datosML/winequality_white.csv", sep = ';')

# COMMAND ----------

# MAGIC %md Se crea una nueva columna y se combinan los datasets

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis = 0)

# Se sustituyen los espacios en blanco en los nombres de las columnas por "_"
data.rename(columns=lambda x : x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md # Visualización de datos
# MAGIC Se exploran los datos con Seaborn y Matplotlib

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize = (25,15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

data.isna().any()

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state = 123)
X_train = train.drop(['quality'], axis = 1)
X_test = test.drop(['quality'], axis = 1)
y_train = train.quality
y_test = test.quality

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time


