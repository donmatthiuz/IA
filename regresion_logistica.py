import numpy as np

class Logistica():
  def __init__(self, learning_rate , epocas):
    self.learning_rate = learning_rate
    self.epocas = epocas

  def delta_j_w(X, Y, w, b):
    #Vamos a calcular el gradiente para nuestro vector w de pesos.
    m = X.shape[0]
    sigmoid = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
    partial_loss = (sigmoid - Y).reshape(-1, 1) * X 
    derivative = partial_loss.mean(axis=0)
    return derivative
  
  def delta_j_b(X, Y, w, b):
    m = X.shape[0]
    sigmoid = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
    partial_loss = sigmoid - Y 
    derivative = partial_loss.mean()
    return derivative

  def algoritmo_sin_librerias(self, X, Y, w,b):
    pass
  def algoritmo_con_librerias(self, dataframe):
    pass
