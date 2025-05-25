import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import time
import random
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional
import warnings
warnings.filterwarnings('ignore')

class BayesianNetworkStructureSearch:
    """
    Algoritmo robusto para búsqueda de estructuras óptimas en redes bayesianas
    """
    
    def __init__(self, data_path_train: str, data_path_test: str, target_var: str = 'target',
                 max_parents: int = 3, cv_folds: int = 5, random_seed: int = 42):
        """
        Inicializa el buscador de estructuras
        
        Args:
            data_path_train: Ruta al archivo de entrenamiento
            data_path_test: Ruta al archivo de prueba
            target_var: Variable objetivo
            max_parents: Máximo número de padres por nodo
            cv_folds: Número de folds para validación cruzada
            random_seed: Semilla para reproducibilidad
        """
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.target_var = target_var
        self.max_parents = max_parents
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        
      
        random.seed(random_seed)
        np.random.seed(random_seed)
        
      
        self.df_train = pd.read_csv(data_path_train).drop(columns=["precipitacion", "year", "mes", "dia"])
        self.df_test = pd.read_csv(data_path_test).drop(columns=["precipitacion", "year", "mes", "dia"])

        
      
        self.continuous_vars = [col for col in self.df_train.columns 
                               if col != target_var and pd.api.types.is_numeric_dtype(self.df_train[col])]
        
        self.all_vars = self.continuous_vars + [target_var]
        
        # Estadísticas iniciales
        self._compute_initial_stats()
        
      
        self.search_results = []
        self.best_structure = None
        self.best_score = -np.inf
        
    def _compute_initial_stats(self):
        """Calcula estadísticas iniciales de los datos"""
      
        target_counts = self.df_train[self.target_var].value_counts()
        total = len(self.df_train)
        self.prior_probs = {val: count / total for val, count in target_counts.items()}
        
        # Estadísticas por variable y target
        self.var_stats = {}
        for var in self.continuous_vars:
            self.var_stats[var] = {}
            for target_val in self.prior_probs.keys():
                subset = self.df_train[self.df_train[self.target_var] == target_val][var]
                self.var_stats[var][target_val] = {
                    'mean': subset.mean(),
                    'std': max(subset.std(), 0.01)  # Evitar std=0
                }
    
    def _is_valid_structure(self, structure: Dict[str, List[str]]) -> bool:
        """
        Verifica si una estructura es válida (DAG sin ciclos)
        
        Args:
            structure: Diccionario con padres de cada nodo
            
        Returns:
            True si es válida, False si no
        """
        # Crear grafo dirigido
        G = nx.DiGraph()
        
        # Añadir nodos
        for node in structure.keys():
            G.add_node(node)
        
        # Añadir aristas
        for child, parents in structure.items():
            for parent in parents:
                if parent in structure or parent == self.target_var:
                    G.add_edge(parent, child)
        
        # Verificar si es DAG (sin ciclos)
        return nx.is_directed_acyclic_graph(G)
    
    def _compute_conditional_likelihood(self, var: str, parents: List[str], 
                                      data: pd.DataFrame) -> float:
        """
        Calcula la verosimilitud condicional de una variable dados sus padres
        
        Args:
            var: Variable objetivo
            parents: Lista de variables padre
            data: Datos para el cálculo
            
        Returns:
            Log-verosimilitud
        """
        if not parents:
            # Sin padres, usar distribución marginal
            if var == self.target_var:
                # Variable discreta
                counts = data[var].value_counts()
                probs = counts / len(data)
                return np.sum([count * np.log(prob + 1e-10) for count, prob in zip(counts, probs)])
            else:
                # Variable continua
                values = data[var].values
                mean, std = values.mean(), max(values.std(), 0.01)
                return np.sum(norm.logpdf(values, mean, std))
        
        total_likelihood = 0
        
        if self.target_var in parents:
            # Depende del target (variable discreta)
            for target_val in self.prior_probs.keys():
                subset = data[data[self.target_var] == target_val]
                if len(subset) == 0:
                    continue
                
                if var == self.target_var:
                    # No debería pasar, pero por si acaso
                    continue
                else:
                    # Variable continua condicional al target
                    values = subset[var].values
                    if len(values) > 0:
                        mean, std = values.mean(), max(values.std(), 0.01)
                        likelihood = np.sum(norm.logpdf(values, mean, std))
                        total_likelihood += likelihood
        else:
            # Dependencias más complejas - simplificación
            # En una implementación completa, calcularíamos todas las combinaciones
            values = data[var].values
            mean, std = values.mean(), max(values.std(), 0.01)
            total_likelihood = np.sum(norm.logpdf(values, mean, std))
        
        return total_likelihood
    
    def _compute_bic_score(self, structure: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """
        Calcula el score BIC para una estructura
        
        Args:
            structure: Estructura de la red
            data: Datos para evaluación
            
        Returns:
            Score BIC (mayor es mejor)
        """
        log_likelihood = 0
        num_parameters = 0
        n_samples = len(data)
        
        for var, parents in structure.items():
            # Calcular likelihood para esta variable
            var_likelihood = self._compute_conditional_likelihood(var, parents, data)
            log_likelihood += var_likelihood
            
            # Contar parámetros
            if var == self.target_var:
                num_parameters += len(self.prior_probs) - 1  # Probabilidades categóricas
            else:
                if self.target_var in parents:
                    num_parameters += 2 * len(self.prior_probs)  # Media y std por cada categoría
                else:
                    num_parameters += 2  # Media y std
        
        # BIC = log-likelihood - (k/2) * log(n)
        bic = log_likelihood - (num_parameters / 2) * np.log(n_samples)
        return bic
    
    def _evaluate_structure_cv(self, structure: Dict[str, List[str]]) -> Dict:
        """
        Evalúa una estructura usando validación cruzada
        
        Args:
            structure: Estructura a evaluar
            
        Returns:
            Diccionario con métricas promedio
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        cv_scores = {
            'bic': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, val_idx in kf.split(self.df_train):
            train_fold = self.df_train.iloc[train_idx]
            val_fold = self.df_train.iloc[val_idx]
            
            # Calcular BIC en fold de entrenamiento
            bic_score = self._compute_bic_score(structure, train_fold)
            cv_scores['bic'].append(bic_score)
            
            # Evaluar predicción en fold de validación
            try:
                pred_metrics = self._evaluate_prediction(structure, train_fold, val_fold)
                cv_scores['accuracy'].append(pred_metrics['accuracy'])
                cv_scores['precision'].append(pred_metrics['precision'])
                cv_scores['recall'].append(pred_metrics['recall'])
                cv_scores['f1'].append(pred_metrics['f1'])
            except:
                # En caso de error, usar valores por defecto
                cv_scores['accuracy'].append(0.5)
                cv_scores['precision'].append(0.5)
                cv_scores['recall'].append(0.5)
                cv_scores['f1'].append(0.5)
        
        # Promediar resultados
        avg_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        avg_scores['std_bic'] = np.std(cv_scores['bic'])
        avg_scores['std_accuracy'] = np.std(cv_scores['accuracy'])
        
        return avg_scores
    
    def _evaluate_prediction(self, structure: Dict[str, List[str]], 
                           train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Evalúa la capacidad predictiva de una estructura
        
        Args:
            structure: Estructura de la red
            train_data: Datos de entrenamiento
            test_data: Datos de prueba
            
        Returns:
            Métricas de predicción
        """
        predictions = []
        true_labels = test_data[self.target_var].values
        
        for _, row in test_data.iterrows():
            # Calcular probabilidades para cada valor del target
            target_probs = {}
            
            for target_val in self.prior_probs.keys():
                prob = self.prior_probs[target_val]  # Prior
                
                # Multiplicar por likelihood de cada variable
                for var in self.continuous_vars:
                    if var in structure:
                        parents = structure[var]
                        if self.target_var in parents:
                            # Usar estadísticas condicionales
                            mean = self.var_stats[var][target_val]['mean']
                            std = self.var_stats[var][target_val]['std']
                        else:
                            # Usar estadísticas marginales
                            all_values = train_data[var]
                            mean, std = all_values.mean(), max(all_values.std(), 0.01)
                        
                        # Calcular likelihood
                        likelihood = norm.pdf(row[var], mean, std)
                        prob *= likelihood
                
                target_probs[target_val] = prob
            
            # Normalizar probabilidades
            total_prob = sum(target_probs.values())
            if total_prob > 0:
                target_probs = {k: v/total_prob for k, v in target_probs.items()}
            
            # Predecir clase con mayor probabilidad
            predicted_class = max(target_probs.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_class)
        
        # Calcular métricas
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def exhaustive_search(self, max_structures: int = 1000):
        """
        Búsqueda exhaustiva de estructuras (para espacios pequeños)
        
        Args:
            max_structures: Máximo número de estructuras a evaluar
        """
        print("Iniciando búsqueda exhaustiva...")
        structures_evaluated = 0
        
        # Generar todas las combinaciones posibles de padres
        for var in self.continuous_vars:
            possible_parents = [self.target_var] + [v for v in self.continuous_vars if v != var]
            
            # Generar todas las combinaciones de padres (hasta max_parents)
            for r in range(min(self.max_parents + 1, len(possible_parents) + 1)):
                for parent_combination in itertools.combinations(possible_parents, r):
                    if structures_evaluated >= max_structures:
                        break
                    
                    # Crear estructura
                    structure = {var: list(parent_combination)}
                    
                    # Verificar validez
                    if self._is_valid_structure(structure):
                        # Evaluar estructura
                        metrics = self._evaluate_structure_cv(structure)
                        
                        self.search_results.append({
                            'structure': structure.copy(),
                            'search_method': 'exhaustive',
                            'metrics': metrics,
                            'complexity': sum(len(parents) for parents in structure.values())
                        })
                        
                        structures_evaluated += 1
                        
                        if structures_evaluated % 100 == 0:
                            print(f"Evaluadas {structures_evaluated} estructuras...")
        
        print(f"Búsqueda exhaustiva completada. {structures_evaluated} estructuras evaluadas.")
    
    def greedy_search(self, max_iterations: int = 500):
        """
        Búsqueda greedy: construye la estructura añadiendo aristas una por una
        
        Args:
            max_iterations: Máximo número de iteraciones
        """
        print("Iniciando búsqueda greedy...")
        
        # Empezar con estructura vacía
        current_structure = {var: [] for var in self.continuous_vars}
        current_score = self._compute_bic_score(current_structure, self.df_train)
        
        for iteration in range(max_iterations):
            best_improvement = 0
            best_addition = None
            
            # Probar añadir cada posible arista
            for var in self.continuous_vars:
                possible_parents = [self.target_var] + [v for v in self.continuous_vars if v != var]
                
                for parent in possible_parents:
                    if parent not in current_structure[var] and len(current_structure[var]) < self.max_parents:
                        # Crear nueva estructura
                        new_structure = {k: v.copy() for k, v in current_structure.items()}
                        new_structure[var].append(parent)
                        
                        # Verificar validez
                        if self._is_valid_structure(new_structure):
                            new_score = self._compute_bic_score(new_structure, self.df_train)
                            improvement = new_score - current_score
                            
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_addition = (var, parent)
            
            # Si encontramos mejora, aplicarla
            if best_improvement > 0:
                var, parent = best_addition
                current_structure[var].append(parent)
                current_score += best_improvement
                print(f"Iteración {iteration + 1}: Añadida arista {parent} -> {var}, "
                      f"Score: {current_score:.2f}")
            else:
                print(f"Búsqueda greedy convergió en iteración {iteration + 1}")
                break
        
        # Evaluar estructura final
        metrics = self._evaluate_structure_cv(current_structure)
        self.search_results.append({
            'structure': current_structure.copy(),
            'search_method': 'greedy',
            'metrics': metrics,
            'complexity': sum(len(parents) for parents in current_structure.values())
        })
        
        print("Búsqueda greedy completada.")
    
    def random_search(self, num_samples: int = 1000):
        """
        Búsqueda aleatoria de estructuras
        
        Args:
            num_samples: Número de estructuras aleatorias a generar
        """
        print("Iniciando búsqueda aleatoria...")
        
        for i in range(num_samples):
            # Generar estructura aleatoria
            structure = {}
            
            for var in self.continuous_vars:
                possible_parents = [self.target_var] + [v for v in self.continuous_vars if v != var]
                
                # Número aleatorio de padres
                num_parents = random.randint(0, min(self.max_parents, len(possible_parents)))
                
                # Seleccionar padres aleatoriamente
                if num_parents > 0:
                    parents = random.sample(possible_parents, num_parents)
                    structure[var] = parents
                else:
                    structure[var] = []
            
            # Verificar validez
            if self._is_valid_structure(structure):
                # Evaluar estructura
                metrics = self._evaluate_structure_cv(structure)
                
                self.search_results.append({
                    'structure': structure.copy(),
                    'search_method': 'random',
                    'metrics': metrics,
                    'complexity': sum(len(parents) for parents in structure.values())
                })
            
            if (i + 1) % 200 == 0:
                print(f"Generadas {i + 1} estructuras aleatorias...")
        
        print("Búsqueda aleatoria completada.")
    
    def hill_climbing_search(self, max_iterations: int = 100, restarts: int = 5):
        """
        Búsqueda por hill climbing con múltiples reinicios
        
        Args:
            max_iterations: Máximo número de iteraciones por reinicio
            restarts: Número de reinicios aleatorios
        """
        print("Iniciando búsqueda por hill climbing...")
        
        for restart in range(restarts):
            print(f"Reinicio {restart + 1}/{restarts}")
            
            # Generar estructura inicial aleatoria
            current_structure = {}
            for var in self.continuous_vars:
                possible_parents = [self.target_var] + [v for v in self.continuous_vars if v != var]
                num_parents = random.randint(0, min(2, len(possible_parents)))
                if num_parents > 0:
                    current_structure[var] = random.sample(possible_parents, num_parents)
                else:
                    current_structure[var] = []
            
            if not self._is_valid_structure(current_structure):
                continue
            
            current_score = self._compute_bic_score(current_structure, self.df_train)
            
            for iteration in range(max_iterations):
                neighbors = self._generate_neighbors(current_structure)
                best_neighbor = None
                best_score = current_score
                
                for neighbor in neighbors:
                    if self._is_valid_structure(neighbor):
                        score = self._compute_bic_score(neighbor, self.df_train)
                        if score > best_score:
                            best_score = score
                            best_neighbor = neighbor
                
                if best_neighbor is not None:
                    current_structure = best_neighbor
                    current_score = best_score
                else:
                    break  # No hay mejora, parar
            
            # Evaluar estructura final de este reinicio
            metrics = self._evaluate_structure_cv(current_structure)
            self.search_results.append({
                'structure': current_structure.copy(),
                'search_method': f'hill_climbing_restart_{restart}',
                'metrics': metrics,
                'complexity': sum(len(parents) for parents in current_structure.values())
            })
        
        print("Búsqueda por hill climbing completada.")
    
    def _generate_neighbors(self, structure: Dict[str, List[str]]) -> List[Dict[str, List[str]]]:
        """
        Genera estructuras vecinas (añadir/quitar una arista)
        
        Args:
            structure: Estructura actual
            
        Returns:
            Lista de estructuras vecinas
        """
        neighbors = []
        
        for var in self.continuous_vars:
            possible_parents = [self.target_var] + [v for v in self.continuous_vars if v != var]
            
            # Quitar una arista
            for parent in structure[var]:
                new_structure = {k: v.copy() for k, v in structure.items()}
                new_structure[var].remove(parent)
                neighbors.append(new_structure)
            
            # Añadir una arista
            for parent in possible_parents:
                if parent not in structure[var] and len(structure[var]) < self.max_parents:
                    new_structure = {k: v.copy() for k, v in structure.items()}
                    new_structure[var].append(parent)
                    neighbors.append(new_structure)
        
        return neighbors
    
    def run_complete_search(self):
        """
        Ejecuta todos los métodos de búsqueda
        """
        print("=" * 80)
        print("INICIANDO BÚSQUEDA COMPLETA DE ESTRUCTURAS DE RED BAYESIANA")
        print("=" * 80)
        print(f"Variables continuas: {self.continuous_vars}")
        print(f"Variable objetivo: {self.target_var}")
        print(f"Máximo padres por nodo: {self.max_parents}")
        print(f"Validación cruzada: {self.cv_folds} folds")
        print()
        
        start_time = time.time()
        
        # Ejecutar diferentes métodos de búsqueda
        self.greedy_search()
        self.random_search(num_samples=800)
        self.hill_climbing_search(restarts=3)
        
        # Si el espacio es pequeño, hacer búsqueda exhaustiva limitada
        if len(self.continuous_vars) <= 4:
            self.exhaustive_search(max_structures=500)
        
        end_time = time.time()
        
        print(f"\nBúsqueda completada en {end_time - start_time:.2f} segundos")
        print(f"Total de estructuras evaluadas: {len(self.search_results)}")
        
        # Encontrar la mejor estructura
        self._find_best_structure()
        
        # Evaluación final en conjunto de prueba
        self._final_evaluation()
    
    def _find_best_structure(self):
        """Encuentra la mejor estructura basada en score compuesto"""
        if not self.search_results:
            print("No hay resultados de búsqueda disponibles.")
            return
        
        # Calcular score compuesto (BIC + accuracy)
        for result in self.search_results:
            bic_score = result['metrics']['bic']
            accuracy = result['metrics']['accuracy']
            f1 = result['metrics']['f1']
            complexity = result['complexity']
            
            # Score compuesto: normalizar y combinar métricas
            # Penalizar por complejidad
            complexity_penalty = complexity * 0.1
            result['composite_score'] = bic_score / 1000 + accuracy + f1 - complexity_penalty
        
        # Ordenar por score compuesto
        self.search_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        self.best_structure = self.search_results[0]
        self.best_score = self.best_structure['composite_score']
    
    def _final_evaluation(self):
        """Evaluación final de la mejor estructura en conjunto de prueba"""
        if not self.best_structure:
            return
        
        print("\n" + "=" * 80)
        print("MEJOR ESTRUCTURA ENCONTRADA")
        print("=" * 80)
        
        structure = self.best_structure['structure']
        metrics = self.best_structure['metrics']
        
        print(f"Método de búsqueda: {self.best_structure['search_method']}")
        print(f"Score compuesto: {self.best_structure['composite_score']:.4f}")
        print(f"Complejidad (número total de aristas): {self.best_structure['complexity']}")
        print()
        
        print("Estructura de la red:")
        for var, parents in structure.items():
            if parents:
                print(f"  {var} <- {', '.join(parents)}")
            else:
                print(f"  {var} (sin padres)")
        print()
        
        print("Métricas de validación cruzada:")
        print(f"  BIC Score: {metrics['bic']:.2f} (±{metrics['std_bic']:.2f})")
        print(f"  Accuracy: {metrics['accuracy']:.4f} (±{metrics['std_accuracy']:.4f})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print()
        
        # Evaluación en conjunto de prueba
        test_metrics = self._evaluate_prediction(structure, self.df_train, self.df_test)
        
        print("Evaluación en conjunto de prueba:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    def get_top_structures(self, top_k: int = 10) -> List[Dict]:
        """
        Obtiene las top-k mejores estructuras
        
        Args:
            top_k: Número de estructuras a retornar
            
        Returns:
            Lista con las mejores estructuras
        """
        return self.search_results[:top_k]
    
    def plot_search_results(self):
        """Visualiza los resultados de la búsqueda"""
        if not self.search_results:
            print("No hay resultados para visualizar.")
            return
        
        # Preparar datos para visualización
        methods = [result['search_method'].split('_')[0] for result in self.search_results]
        accuracies = [result['metrics']['accuracy'] for result in self.search_results]
        bic_scores = [result['metrics']['bic'] for result in self.search_results]
        complexities = [result['complexity'] for result in self.search_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy por método
        axes[0, 0].boxplot([acc for acc, method in zip(accuracies, methods) if method == 'greedy'], 
                          labels=['Greedy'])
        axes[0, 0].set_title('Accuracy por Método de Búsqueda')
        axes[0, 0].set_ylabel('Accuracy')
        
        # BIC vs Accuracy
        axes[0, 1].scatter(bic_scores, accuracies, alpha=0.6, c=complexities, cmap='viridis')
        axes[0, 1].set_xlabel('BIC Score')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('BIC Score vs Accuracy (Color = Complejidad)')
        
        # Distribución de complejidades
        axes[1, 0].hist(complexities, bins=20, alpha=0.7)
        axes[1, 0].set_xlabel('Complejidad (Número de Aristas)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].set_title('Distribución de Complejidades')
        
        # Top 10 estructuras
        top_10 = self.get_top_structures(10)
        top_scores = [struct['composite_score'] for struct in top_10]
        top_methods = [struct['search_method'].split('_')[0] for struct in top_10]
        
        axes[1, 1].bar(range(len(top_scores)), top_scores, 
                      color=['red' if m == 'greedy' else 'blue' if m == 'random' else 'green' 
                            for m in top_methods])
        axes[1, 1].set_xlabel('Ranking')
        axes[1, 1].set_ylabel('Score Compuesto')
        axes[1, 1].set_title('Top 10 Estructuras')
        
        plt.tight_layout()
        plt.show()


# Función principal para usar el algoritmo
def main():
    """Función principal para ejecutar la búsqueda de estructuras"""
    
    # Configurar rutas de archivos
    train_path = "../data/balanceado_train.csv"
    test_path = "../data/balanceado_test.csv"
    
    # Inicializar buscador
    searcher = BayesianNetworkStructureSearch(
        data_path_train=train_path,
        data_path_test=test_path,
        target_var='target',
        max_parents=3,
        cv_folds=5,
        random_seed=42
    )
    
    # Ejecutar búsqueda completa
    searcher.run_complete_search()
    
    # Mostrar top 5 estructuras
    print("\n" + "=" * 80)
    print("TOP 5 ESTRUCTURAS")
    print("=" * 80)
    
    top_structures = searcher.get_top_structures(5)
    for i, struct in enumerate(top_structures, 1):
        print(f"\n{i}. Score: {struct['composite_score']:.4f} "
              f"(Método: {struct['search_method']}, "
              f"Accuracy: {struct['metrics']['accuracy']:.4f})")
        
        for var, parents in struct['structure'].items():
            if parents:
                print(f"   {var} <- {', '.join(parents)}")
            else:
                print(f"   {var} (sin padres)")
    
    # Visualizar resultados (opcional)
    try:
        searcher.plot_search_results()
    except:
        print("No se pudo generar la visualización (matplotlib no disponible)")
    
    return searcher


class AdvancedBayesianStructureSearch(BayesianNetworkStructureSearch):
    """
    Versión avanzada con algoritmos adicionales y optimizaciones
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabu_list = []
        self.tabu_size = 50
        
    def simulated_annealing_search(self, initial_temp: float = 100.0, 
                                 cooling_rate: float = 0.95, 
                                 min_temp: float = 0.01,
                                 max_iterations: int = 1000):
        """
        Búsqueda por recocido simulado (Simulated Annealing)
        
        Args:
            initial_temp: Temperatura inicial
            cooling_rate: Tasa de enfriamiento
            min_temp: Temperatura mínima
            max_iterations: Máximo número de iteraciones
        """
        print("Iniciando búsqueda por recocido simulado...")
        
        # Generar estructura inicial aleatoria
        current_structure = self._generate_random_structure()
        current_score = self._compute_bic_score(current_structure, self.df_train)
        
        best_structure = current_structure.copy()
        best_score = current_score
        
        temperature = initial_temp
        iteration = 0
        
        while temperature > min_temp and iteration < max_iterations:
            # Generar estructura vecina
            neighbors = self._generate_neighbors(current_structure)
            if not neighbors:
                break
                
            new_structure = random.choice(neighbors)
            
            if self._is_valid_structure(new_structure):
                new_score = self._compute_bic_score(new_structure, self.df_train)
                
                # Calcular probabilidad de aceptación
                delta = new_score - current_score
                
                if delta > 0 or random.random() < np.exp(delta / temperature):
                    current_structure = new_structure
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_structure = current_structure.copy()
                        best_score = current_score
            
            # Enfriar
            temperature *= cooling_rate
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"Iteración {iteration}, Temperatura: {temperature:.4f}, "
                      f"Mejor score: {best_score:.2f}")
        
        # Evaluar mejor estructura encontrada
        metrics = self._evaluate_structure_cv(best_structure)
        self.search_results.append({
            'structure': best_structure.copy(),
            'search_method': 'simulated_annealing',
            'metrics': metrics,
            'complexity': sum(len(parents) for parents in best_structure.values())
        })
        
        print("Búsqueda por recocido simulado completada.")
    
    def tabu_search(self, max_iterations: int = 500, tabu_size: int = 50):
        """
        Búsqueda tabú
        
        Args:
            max_iterations: Máximo número de iteraciones
            tabu_size: Tamaño de la lista tabú
        """
        print("Iniciando búsqueda tabú...")
        
        self.tabu_size = tabu_size
        self.tabu_list = []
        
        # Estructura inicial
        current_structure = self._generate_random_structure()
        best_structure = current_structure.copy()
        best_score = self._compute_bic_score(current_structure, self.df_train)
        
        for iteration in range(max_iterations):
            neighbors = self._generate_neighbors(current_structure)
            
            # Filtrar vecinos que no estén en lista tabú
            valid_neighbors = []
            for neighbor in neighbors:
                if self._is_valid_structure(neighbor):
                    neighbor_key = self._structure_to_key(neighbor)
                    if neighbor_key not in self.tabu_list:
                        valid_neighbors.append(neighbor)
            
            if not valid_neighbors:
                # Si no hay vecinos válidos, reiniciar
                current_structure = self._generate_random_structure()
                continue
            
            # Encontrar el mejor vecino
            best_neighbor = None
            best_neighbor_score = -np.inf
            
            for neighbor in valid_neighbors:
                score = self._compute_bic_score(neighbor, self.df_train)
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor
            
            # Actualizar estructura actual
            current_structure = best_neighbor
            current_key = self._structure_to_key(current_structure)
            
            # Añadir a lista tabú
            self.tabu_list.append(current_key)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)
            
            # Actualizar mejor solución si es necesario
            if best_neighbor_score > best_score:
                best_structure = current_structure.copy()
                best_score = best_neighbor_score
                print(f"Iteración {iteration}: Nueva mejor solución, score: {best_score:.2f}")
        
        # Evaluar mejor estructura
        metrics = self._evaluate_structure_cv(best_structure)
        self.search_results.append({
            'structure': best_structure.copy(),
            'search_method': 'tabu_search',
            'metrics': metrics,
            'complexity': sum(len(parents) for parents in best_structure.values())
        })
        
        print("Búsqueda tabú completada.")
    
    def genetic_algorithm_search(self, population_size: int = 50, 
                               generations: int = 100,
                               mutation_rate: float = 0.1,
                               crossover_rate: float = 0.8):
        """
        Algoritmo genético para búsqueda de estructuras
        
        Args:
            population_size: Tamaño de la población
            generations: Número de generaciones
            mutation_rate: Tasa de mutación
            crossover_rate: Tasa de cruzamiento
        """
        print("Iniciando algoritmo genético...")
        
        # Generar población inicial
        population = []
        for _ in range(population_size):
            structure = self._generate_random_structure()
            if self._is_valid_structure(structure):
                population.append(structure)
        
        for generation in range(generations):
            # Evaluar población
            fitness_scores = []
            for structure in population:
                score = self._compute_bic_score(structure, self.df_train)
                fitness_scores.append(score)
            
            # Selección por torneo
            new_population = []
            for _ in range(population_size):
                # Seleccionar padres
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Cruzamiento
                if random.random() < crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutación
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                # Verificar validez
                if self._is_valid_structure(child):
                    new_population.append(child)
                else:
                    new_population.append(self._generate_random_structure())
            
            population = new_population
            
            if generation % 20 == 0:
                best_fitness = max(fitness_scores)
                print(f"Generación {generation}: Mejor fitness = {best_fitness:.2f}")
        
        # Encontrar la mejor estructura de la población final
        final_fitness = [self._compute_bic_score(struct, self.df_train) for struct in population]
        best_idx = np.argmax(final_fitness)
        best_structure = population[best_idx]
        
        # Evaluar mejor estructura
        metrics = self._evaluate_structure_cv(best_structure)
        self.search_results.append({
            'structure': best_structure.copy(),
            'search_method': 'genetic_algorithm',
            'metrics': metrics,
            'complexity': sum(len(parents) for parents in best_structure.values())
        })
        
        print("Algoritmo genético completado.")
    
    def _generate_random_structure(self) -> Dict[str, List[str]]:
        """Genera una estructura aleatoria válida"""
        structure = {}
        for var in self.continuous_vars:
            possible_parents = [self.target_var] + [v for v in self.continuous_vars if v != var]
            num_parents = random.randint(0, min(self.max_parents, len(possible_parents)))
            
            if num_parents > 0:
                parents = random.sample(possible_parents, num_parents)
                structure[var] = parents
            else:
                structure[var] = []
        
        return structure
    
    def _structure_to_key(self, structure: Dict[str, List[str]]) -> str:
        """Convierte una estructura a string para usar como clave"""
        key_parts = []
        for var in sorted(structure.keys()):
            parents = sorted(structure[var])
            key_parts.append(f"{var}:{','.join(parents)}")
        return "|".join(key_parts)
    
    def _tournament_selection(self, population: List, fitness_scores: List, 
                            tournament_size: int = 3):
        """Selección por torneo para algoritmo genético"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Cruzamiento entre dos estructuras"""
        child = {}
        for var in self.continuous_vars:
            # Elegir aleatoriamente de cuál padre heredar
            if random.random() < 0.5:
                child[var] = parent1[var].copy()
            else:
                child[var] = parent2[var].copy()
        return child
    
    def _mutate(self, structure: Dict) -> Dict:
        """Mutación de una estructura"""
        mutated = {k: v.copy() for k, v in structure.items()}
        
        # Elegir variable aleatoria para mutar
        var_to_mutate = random.choice(self.continuous_vars)
        possible_parents = [self.target_var] + [v for v in self.continuous_vars if v != var_to_mutate]
        
        # Tipo de mutación aleatoria
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and len(mutated[var_to_mutate]) < self.max_parents:
            # Añadir padre
            available_parents = [p for p in possible_parents if p not in mutated[var_to_mutate]]
            if available_parents:
                new_parent = random.choice(available_parents)
                mutated[var_to_mutate].append(new_parent)
        
        elif mutation_type == 'remove' and mutated[var_to_mutate]:
            # Quitar padre
            parent_to_remove = random.choice(mutated[var_to_mutate])
            mutated[var_to_mutate].remove(parent_to_remove)
        
        elif mutation_type == 'replace' and mutated[var_to_mutate]:
            # Reemplazar padre
            old_parent = random.choice(mutated[var_to_mutate])
            available_parents = [p for p in possible_parents if p not in mutated[var_to_mutate]]
            if available_parents:
                new_parent = random.choice(available_parents)
                mutated[var_to_mutate].remove(old_parent)
                mutated[var_to_mutate].append(new_parent)
        
        return mutated
    
    def run_advanced_search(self):
        """Ejecuta búsqueda con algoritmos avanzados"""
        print("=" * 80)
        print("INICIANDO BÚSQUEDA AVANZADA DE ESTRUCTURAS")
        print("=" * 80)
        
        start_time = time.time()
        
        # Ejecutar algoritmos básicos
        self.greedy_search()
        self.random_search(num_samples=500)
        self.hill_climbing_search(restarts=3)
        
        # Ejecutar algoritmos avanzados
        self.simulated_annealing_search(max_iterations=800)
        self.tabu_search(max_iterations=400)
        self.genetic_algorithm_search(population_size=30, generations=50)
        
        end_time = time.time()
        
        print(f"\nBúsqueda avanzada completada en {end_time - start_time:.2f} segundos")
        print(f"Total de estructuras evaluadas: {len(self.search_results)}")
        
        # Análisis de resultados
        self._find_best_structure()
        self._final_evaluation()
        self._analyze_search_methods()
    
    def _analyze_search_methods(self):
        """Analiza el rendimiento de diferentes métodos de búsqueda"""
        print("\n" + "=" * 80)
        print("ANÁLISIS DE MÉTODOS DE BÚSQUEDA")
        print("=" * 80)
        
        method_performance = defaultdict(list)
        
        for result in self.search_results:
            method = result['search_method'].split('_')[0]
            method_performance[method].append(result['composite_score'])
        
        print("Rendimiento promedio por método:")
        for method, scores in method_performance.items():
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            best_score = max(scores)
            print(f"  {method:20s}: {avg_score:.4f} (±{std_score:.4f}) "
                  f"[mejor: {best_score:.4f}] ({len(scores)} estructuras)")
        
        # Encontrar mejor estructura de cada método
        print("\nMejor estructura por método:")
        best_by_method = {}
        for result in self.search_results:
            method = result['search_method'].split('_')[0]
            if method not in best_by_method or result['composite_score'] > best_by_method[method]['composite_score']:
                best_by_method[method] = result
        
        for method, result in best_by_method.items():
            print(f"  {method:20s}: Score {result['composite_score']:.4f}, "
                  f"Accuracy {result['metrics']['accuracy']:.4f}")


# Función para ejecutar la búsqueda avanzada
def run_advanced_structure_search():
    """Función para ejecutar la búsqueda avanzada"""
    
    # Configurar rutas
    train_path = "../data/balanceado_train.csv"
    test_path = "../data/balanceado_test.csv"
    
    # Crear buscador avanzado
    advanced_searcher = AdvancedBayesianStructureSearch(
        data_path_train=train_path,
        data_path_test=test_path,
        target_var='target',
        max_parents=3,
        cv_folds=5,
        random_seed=42
    )
    
    # Ejecutar búsqueda avanzada
    advanced_searcher.run_advanced_search()
    
    return advanced_searcher


if __name__ == "__main__":
    # Ejecutar búsqueda básica
    print("Ejecutando búsqueda básica...")
    basic_searcher = main()
    
    print("\n" + "="*100 + "\n")
    
    # Ejecutar búsqueda avanzada
    print("Ejecutando búsqueda avanzada...")
    advanced_searcher = run_advanced_structure_search()