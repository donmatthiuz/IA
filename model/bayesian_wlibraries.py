import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
from itertools import combinations
from collections import deque
import math

class BreadthFirstSearch:
    def __init__(self, data, max_edges=None):
        self.data = data
        self.variables = list(data.columns)
        self.max_edges = max_edges or min(len(self.variables) * 2, 15)
        self.best_score = -float('inf')
        self.best_model = None
    
    def _calculate_bic_score(self, model):
        """Implementación manual del BIC Score"""
        try:
            # Entrenar el modelo
            model.fit(self.data, estimator=MaximumLikelihoodEstimator)
            
            # Calcular log-likelihood
            log_likelihood = 0
            n_samples = len(self.data)
            
            for node in model.nodes():
                parents = list(model.predecessors(node))
                
                if not parents:
                    # Nodo sin padres
                    counts = self.data[node].value_counts()
                    probs = counts / n_samples
                    log_likelihood += np.sum(counts * np.log(probs + 1e-10))
                else:
                    # Nodo con padres
                    parent_combinations = self.data[parents + [node]].groupby(parents + [node]).size()
                    parent_totals = self.data[parents].groupby(parents).size()
                    
                    for idx, count in parent_combinations.items():
                        parent_idx = idx[:-1] if len(parents) > 1 else idx[0]
                        prob = count / parent_totals[parent_idx]
                        log_likelihood += count * math.log(prob + 1e-10)
            
            # Calcular número de parámetros
            n_params = 0
            for node in model.nodes():
                parents = list(model.predecessors(node))
                node_states = len(self.data[node].unique())
                
                if not parents:
                    n_params += node_states - 1
                else:
                    parent_states = 1
                    for parent in parents:
                        parent_states *= len(self.data[parent].unique())
                    n_params += parent_states * (node_states - 1)
            
            # BIC = log_likelihood - (k/2) * log(n)
            bic_score = log_likelihood - (n_params / 2) * math.log(n_samples)
            return bic_score
            
        except Exception as e:
            return -float('inf')
    
    def _creates_cycle(self, graph, parent, child):
        """Verifica si agregar un arco crearía un ciclo"""
        temp_graph = graph.copy()
        temp_graph.add_edge(parent, child)
        return not nx.is_directed_acyclic_graph(temp_graph)
    
    def _graph_to_tuple(self, graph):
        """Convierte un grafo a tupla para poder usar como key en set"""
        return tuple(sorted(graph.edges()))
    
    def _get_possible_actions(self, graph):
        """Obtiene todas las posibles acciones (agregar arcos) desde el estado actual"""
        actions = []
        for parent, child in combinations(self.variables, 2):
            # Probar ambas direcciones
            for p, c in [(parent, child), (child, parent)]:
                if not graph.has_edge(p, c) and not self._creates_cycle(graph, p, c):
                    actions.append((p, c))
        return actions
    
    def _apply_action(self, graph, action):
        """Aplica una acción (agregar arco) al grafo"""
        new_graph = graph.copy()
        parent, child = action
        new_graph.add_edge(parent, child)
        return new_graph
    
    def _is_goal_state(self, graph):
        """Determina si el estado actual es un estado objetivo"""
        # El objetivo es encontrar el mejor modelo, por lo que evaluamos cada estado
        if len(graph.edges()) == 0:
            return False
        
        try:
            model = BayesianNetwork(graph.edges())
            score = self._calculate_bic_score(model)
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                print(f"Nuevo mejor modelo encontrado! Score: {score:.3f}, Arcos: {len(graph.edges())}")
                return True
        except:
            pass
        return False
    
    def breadth_first_search(self):
        """Implementación de BFS para aprendizaje de estructura"""
        print(f"Iniciando Breadth First Search (máximo {self.max_edges} arcos)")
        
        # Estado inicial: grafo vacío
        initial_state = nx.DiGraph()
        initial_state.add_nodes_from(self.variables)
        
        # Cola para BFS: (estado_actual, camino)
        queue = deque([(initial_state, [])])
        visited = set()
        
        nodes_explored = 0
        
        while queue:
            current_graph, path = queue.popleft()
            nodes_explored += 1
            
            # Convertir grafo a representación hasheable
            graph_key = self._graph_to_tuple(current_graph)
            
            if graph_key in visited:
                continue
                
            visited.add(graph_key)
            
            # Verificar si es estado objetivo (actualiza best_model si es mejor)
            self._is_goal_state(current_graph)
            
            # Si hemos alcanzado el máximo de arcos, no expandir más
            if len(current_graph.edges()) >= self.max_edges:
                continue
            
            # Expandir: generar todos los sucesores válidos
            actions = self._get_possible_actions(current_graph)
            
            for action in actions:
                next_graph = self._apply_action(current_graph, action)
                next_key = self._graph_to_tuple(next_graph)
                
                if next_key not in visited:
                    new_path = path + [current_graph]
                    queue.append((next_graph, new_path))
            
            # Progreso cada 1000 nodos explorados
            if nodes_explored % 1000 == 0:
                print(f"Nodos explorados: {nodes_explored}, Cola: {len(queue)}, Visitados: {len(visited)}")
        
        print(f"BFS completado. Nodos explorados: {nodes_explored}")
        
        # Retornar el mejor modelo encontrado
        if self.best_model is not None:
            print(f"Mejor modelo encontrado con score: {self.best_score:.3f}")
            print(f"Arcos: {list(self.best_model.edges())}")
            return self.best_model
        else:
            print("No se encontró estructura válida, usando grafo vacío")
            empty_model = BayesianNetwork()
            empty_model.add_nodes_from(self.variables)
            return empty_model

# Carga los datos
print("Cargando datos...")
data = pd.read_csv("../data/balanceado.csv")

# Codificar variables categóricas
label_enc = LabelEncoder()
data['franja_horaria'] = label_enc.fit_transform(data['franja_horaria'])

# Discretizar variables continuas
features_to_discretize = ['temperatura', 'humedad', 'viento_vel_m_s', 'viento_dir',
                          'presion', 'precipitacion', 'nubosidad']
disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
data[features_to_discretize] = disc.fit_transform(data[features_to_discretize])

# Convertir a enteros
for col in data.columns:
    data[col] = data[col].astype(int)

# Eliminar columnas no necesarias
data = data.drop(columns=['year', 'mes', 'dia'])

print(f"Datos procesados: {data.shape}")
print(f"Variables: {list(data.columns)}")

# Separar en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Aprender la estructura con BFS
bfs = BreadthFirstSearch(train_data, max_edges=8)  # Limitar para eficiencia
best_model = bfs.breadth_first_search()

print("\nMejor estructura encontrada:")
print("Arcos:", list(best_model.edges()))

if len(best_model.edges()) > 0:
    # Ajustar los parámetros con Maximum Likelihood
    print("Entrenando parámetros...")
    best_model.fit(train_data, estimator=MaximumLikelihoodEstimator)
    
    # Inferencia
    inference = VariableElimination(best_model)
    
    # Predecir target en datos de prueba
    print("Realizando predicciones...")
    y_true = test_data['target'].values
    y_pred = []
    
    for idx, row in test_data.iterrows():
        evidence = {k: v for k, v in row.items() if k != 'target'}
        try:
            pred = inference.map_query(variables=['target'], evidence=evidence)
            y_pred.append(pred['target'])
        except:
            # En caso de error, usar clase mayoritaria
            y_pred.append(int(train_data['target'].mode()[0]))
    
    # Evaluar precisión
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Precisión en test con BFS: {accuracy:.3f}")
else:
    print("No se pudo aprender ninguna estructura válida")
    # Usar predicción por clase mayoritaria
    majority_class = int(train_data['target'].mode()[0])
    y_pred = [majority_class] * len(test_data)
    accuracy = accuracy_score(test_data['target'].values, y_pred)
    print(f"Precisión con clase mayoritaria: {accuracy:.3f}")

print(f"Total de predicciones: {len(y_pred) if 'y_pred' in locals() else 0}")
print("Proceso completado.")