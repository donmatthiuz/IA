import numpy as np
from sklearn import svm

class SVM ():
  def __init__(self, learning_rate , epocas, lambda_par):
    self.learning_rate = learning_rate
    self.epocas = epocas
    self.lambda_par = lambda_par
    self.w = []
    self.b = 0

  def delta_j_w_1(self, w):
    return self.learning_rate * (2 * self.lambda_par * w)
  
  def delta_j_w_2(self, w, x_i, y_i):
    return self.learning_rate * ( 2 * self.lambda_par *  w - np.dot(x_i, y_i) )
  
  def algoritmo_sin_librerias_entrenamiento(self, X, Y, X_test):
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
          self.w -= self.delta_j_w_2(x_i, y[j], self.w, self.b)
          self.b -=  y[j] * self.learning_rate
    
    y_predichas = np.dot(X_test, self.w) - self.b
    return np.sign(y_predichas)
  

  def algoritmo_con_librerias(self, X_train, Y_train, X_test):
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)


  