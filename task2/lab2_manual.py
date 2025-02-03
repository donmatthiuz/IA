import pandas as pd
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Utilizamos solamente sklearn para repartir el dataset y para evaluar las métricas de prueba

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

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        # Laplace 
        self.alpha = alpha
        self.class_probs = {}
        self.word_probs = {"spam": defaultdict(lambda: self.alpha), "ham": defaultdict(lambda: self.alpha)}
        self.total_words = {"spam": 0, "ham": 0}
        self.vocab = set()

    def train(self, X, y):
        class_counts = {"spam": 0, "ham": 0}

        for message, label in zip(X, y):
            class_counts[label] += 1
            words = re.findall(r"\b\w+\b", message)
            for word in words:
                self.word_probs[label][word] += 1
                self.total_words[label] += 1
                self.vocab.add(word)

        total_messages = len(y)
        self.class_probs["spam"] = class_counts["spam"] / total_messages
        self.class_probs["ham"] = class_counts["ham"] / total_messages

    def predict(self, message):
        words = re.findall(r"\b\w+\b", message)
        spam_prob = self.class_probs["spam"]
        ham_prob = self.class_probs["ham"]
        
        for word in words:
            spam_prob *= self.word_probs["spam"][word] / (self.total_words["spam"] + len(self.vocab))
            ham_prob *= self.word_probs["ham"][word] / (self.total_words["ham"] + len(self.vocab))

        return "spam" if spam_prob > ham_prob else "ham"

# Entrenamos el modelo
nb_manual = NaiveBayesClassifier()
nb_manual.train(X_train, y_train)

# Lo evaluamos con el set de prueba
predictions = [nb_manual.predict(msg) for msg in X_test]

# Para las métricas, utlizamos precisión, recall y fscore debido a que el set dataset está desbalanceado
# Hay solamente 748 mensajes de spam mientras que hay 4819 mensajes de ham
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="weighted")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
