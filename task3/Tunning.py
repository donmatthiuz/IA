import numpy as np
from sklearn.metrics import accuracy_score
from SVM import SVM_manual

class Tunning ():
  def __init__(self, X_val , y_val, X_Train, Y_train):
    self.learning_rates = [0.0001, 0.001, 0.01]
    self.epocas_options = [500, 1000, 2000]
    self.X_val = X_val
    self.y_val =  np.where( y_val <= 0, -1, 1)
    self.X_train = X_Train
    self.y_train = Y_train
    self.lambda_options = [0.001, 0.01, 0.1]
    self.best_acc = 0
    self.best_params = None
    self.best_model = None

    self.param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
  def encontrar_variables(self):
    for lr in self.learning_rates:
      for ep in self.epocas_options:
          for lamb in self.lambda_options:

              model = SVM_manual(learning_rate=lr, epocas=ep, lambda_par=lamb)
              model.algoritmo_sin_librerias_fit(X=self.X_train, Y=self.y_train)

              y_pred = model.algoritmo_sin_librerias_prediccion(X_test=self.X_val)
              acc = accuracy_score(self.y_val, y_pred)

              print(f"Tuning Manual - lr: {lr}, ep: {ep}, lambda: {lamb} -> Acc: {acc:.4f}")

              # Guardar el mejor modelo
              if acc > self.best_acc:
                  self.best_acc = acc
                  self.best_params = (lr, ep, lamb)
                  self.best_model = model