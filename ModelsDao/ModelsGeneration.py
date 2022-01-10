import pandas as pd
import numpy as np
from scipy.sparse.construct import random
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR,SVC


def buildLinearModel(X,y):
  linear_model = LinearRegression()
  linear_model.fit(X,y)
  return linear_model

def buildLogisticRegressionModel(X,y):
  logistic_model = LogisticRegression()
  logistic_model.fit(X,y)
  return logistic_model

def buildKNNModel_Reg(X,y,k):
  knn_model = KNeighborsRegressor(n_neighbors=k)
  knn_model.fit(X,y)
  return knn_model

def buildKNNModel_Cls(X,y,k):
  knn_model = KNeighborsRegressor(n_neighbors=k)
  knn_model.fit(X,y)
  return knn_model







