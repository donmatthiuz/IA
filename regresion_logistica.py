import numpy as np
from sklearn.linear_model import LogisticRegression

class Logistica():
  def __init__(self, learning_rate , epocas):
    self.learning_rate = learning_rate
    self.epocas = epocas

  def delta_j_w(self, X, Y, w, b):
    #Vamos a calcular el gradiente para nuestro vector w de pesos.
    m = X.shape[0]
    sigmoid = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
    partial_loss = (sigmoid - Y).reshape(-1, 1) * X 
    derivative = partial_loss.mean(axis=0)
    return derivative
  
  def delta_j_b(self, X, Y, w, b):
    m = X.shape[0]
    sigmoid = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
    partial_loss = sigmoid - Y 
    derivative = partial_loss.mean()
    return derivative

  def algoritmo_sin_librerias(self, X, Y, X_test, w,b):
    for epoch in range(self.epocas):
      dw = self.delta_j_w(X, Y, w, b)
      db = self.delta_j_b(X, Y, w, b)
      w -= self.learning_rate * dw
      b -= self.learning_rate * db
      sigmoid = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
      loss = -np.mean(Y * np.log(sigmoid) + (1 - Y) * np.log(1 - sigmoid))
      if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
    y_prediccion = 1 / (1 + np.exp(-(np.dot(X_test, w) + b)))
    y_predichas = [1 if val >= 0.5 else 0 for val in y_prediccion]
    return y_predichas

    pass
  def algoritmo_con_librerias(self, X_train, Y_train, X_test):
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, Y_train)
    y_pred = logreg.predict(X_test)
    return y_pred



