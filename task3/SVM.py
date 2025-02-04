import numpy as np


class SVM_manual ():
  def __init__(self, learning_rate , epocas, lambda_par):
    self.learning_rate = learning_rate
    self.epocas = epocas
    self.lambda_par = lambda_par
    self.w = None
    self.b = None

  def delta_j_w_1(self, w):
    return self.learning_rate * (2 * self.lambda_par * w)
  
  def delta_j_w_2(self, w, x_i, y_i):
    return self.learning_rate * (2 * self.lambda_par * w - np.dot(x_i, y_i))
  
  def algoritmo_sin_librerias_fit(self, X, Y):
    _, n_s = X.shape

    y = np.where(Y <= 0, -1, 1)
    self.w = np.zeros(n_s) #llenamos el vector de pesos con 0's
    self.b = 0

    for i in range(self.epocas):
       for j, x_i in enumerate(X):
        condition = y[j] * (np.dot(x_i, self.w) - self.b) >= 1
        if condition:
          self.w -= self.delta_j_w_1(self.w)
        else:
          self.w -= self.delta_j_w_2(w=self.w, x_i=x_i, y_i=y[j])
          self.b -=  self.learning_rate * y[j]  
       if i % 100 == 0:
         print(f"Epoca actual {i}")
  
  def algoritmo_sin_librerias_prediccion(self, X_test):
     y_predichas = np.dot(X_test, self.w) - self.b
     return np.sign(y_predichas)

    
