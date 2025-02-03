import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support

file_path = "task2\\entrenamiento.txt"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t") 
        if len(parts) == 2:
            label, message = parts
            data.append((label, message)) 

df = pd.DataFrame(data, columns=["label", "message"])

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train) 
X_test_vec = vectorizer.transform(X_test)  

nb_model = MultinomialNB(alpha=1.0)  
nb_model.fit(X_train_vec, y_train)

test_predictions = nb_model.predict(X_test_vec)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predictions, average="weighted")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Manual

# Precision: 0.9817
# Recall: 0.9811
# F1 Score: 0.9813

# Librerías

# Precision: 0.9800
# Recall: 0.9802
# F1 Score: 0.9801

# La diferencia es casi nula, pero el modelo manual sobrepasa al modelo con librerías por una cantidad aproximada
# de 0.0010. Esto puede deverse a la diferencia entre la tokenización de nuestro modelo comparado al 
# CountVectorizer() de scikit-learn. O tal vez debido a alguna diferencia en la implementación de Laplace 
# comparado con el de scikit. Aunque, podría ser también algo tan simple como alguna optimización a
# los cálculos de números decimales que implemente scikit que cause la pequeña discrepacia. En general, 
# ambos modelos están bastante similares en performance.