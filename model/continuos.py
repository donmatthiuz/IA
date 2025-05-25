import math
import random

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    
    def add_node(self, name, states):
        """Agregar un nodo a la red con sus posibles estados"""
        self.nodes[name] = {
            'states': states,
            'parents': [],
            'children': [],
            'cpt': {},
            'type': 'discrete'
        }
        self.edges[name] = []
    
    def add_continuous_node(self, name, parents=None):
        """Agregar un nodo continuo que sigue una distribución gaussiana"""
        if parents is None:
            parents = []
        
        self.nodes[name] = {
            'states': 'continuous',
            'parents': parents,
            'children': [],
            'cpt': {},
            'type': 'continuous',
            'gaussian_params': {}  # Almacenará mu y sigma para cada combinación de padres
        }
        self.edges[name] = []
        
        # Actualizar las relaciones padre-hijo
        for parent in parents:
            if parent in self.nodes:
                self.nodes[parent]['children'].append(name)
                self.edges[parent].append(name)
    
    def add_edge(self, parent, child):
        """Agregar una arista dirigida de parent a child"""
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Ambos nodos deben existir antes de agregar la arista")
        
        self.nodes[parent]['children'].append(child)
        self.nodes[child]['parents'].append(parent)
        self.edges[parent].append(child)
    
    def set_cpt(self, node, cpt):
        if node not in self.nodes:
            raise ValueError(f"El nodo {node} no existe")
        
        for condition, probs in cpt.items():
            if abs(sum(probs.values()) - 1.0) > 1e-6:
                raise ValueError(f"Las probabilidades para {condition} no suman 1")
        
        self.nodes[node]['cpt'] = cpt
    
    def set_gaussian_params(self, node, params):
        if node not in self.nodes:
            raise ValueError(f"El nodo {node} no existe")
        
        if self.nodes[node]['type'] != 'continuous':
            raise ValueError(f"El nodo {node} no es continuo")
        
        for condition, param in params.items():
            if param['sigma'] == 0:
                print(f"Advertencia: sigma cero detectado para condición {condition} en nodo {node}, asignando valor pequeño")
                param['sigma'] = 1e-6  # corregir sigma cero
        
        self.nodes[node]['gaussian_params'] = params

    
    def gaussian_pdf(self, x, mu, sigma):
        if sigma <= 0:
            raise ValueError(f"Sigma debe ser mayor que 0, recibido sigma={sigma}")
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def get_probability(self, node, value, evidence=None):
        if evidence is None:
            evidence = {}
        
        node_info = self.nodes[node]
        
        if node_info['type'] == 'continuous':
            return self.get_continuous_probability(node, value, evidence)
        
        cpt = node_info['cpt']
        parents = node_info['parents']
        
        if not parents:
            return cpt.get((), {}).get(value, 0)
        else:
            condition = tuple(evidence.get(parent, None) for parent in parents)
            return cpt.get(condition, {}).get(value, 0)
    
    def get_continuous_probability(self, node, value, evidence=None):
        """Obtener la densidad de probabilidad para un nodo continuo"""
        if evidence is None:
            evidence = {}
        
        node_info = self.nodes[node]
        parents = node_info['parents']
        gaussian_params = node_info['gaussian_params']
        
        if not parents:
            # Nodo raíz continuo
            params = gaussian_params.get((), {'mu': 0, 'sigma': 1})
        else:
            # Construir la condición basada en los padres
            condition = tuple(evidence.get(parent, None) for parent in parents)
            params = gaussian_params.get(condition, {'mu': 0, 'sigma': 1})
        
        return self.gaussian_pdf(value, params['mu'], params['sigma'])
    
    def enumerate_all(self, vars_list, evidence):
        if not vars_list:
            return 1.0
        
        first_var = vars_list[0]
        rest_vars = vars_list[1:]
        
        if first_var in evidence:
            prob = self.get_probability(first_var, evidence[first_var], evidence)
            return prob * self.enumerate_all(rest_vars, evidence)
        else:
            if self.nodes[first_var]['type'] == 'continuous':
                # Para nodos continuos, necesitamos una aproximación
                # Usaremos sampling o integración numérica
                return self.approximate_continuous_enumeration(first_var, rest_vars, evidence)
            else:
                total = 0.0
                for value in self.nodes[first_var]['states']:
                    new_evidence = evidence.copy()
                    new_evidence[first_var] = value
                    prob = self.get_probability(first_var, value, new_evidence)
                    total += prob * self.enumerate_all(rest_vars, new_evidence)
                return total
    
    def approximate_continuous_enumeration(self, continuous_var, rest_vars, evidence):
        """Aproximar la enumeración para variables continuas usando sampling"""
        total = 0.0
        num_samples = 100  # Número de muestras para la aproximación
        
        # Obtener parámetros gaussianos para el nodo continuo
        node_info = self.nodes[continuous_var]
        parents = node_info['parents']
        gaussian_params = node_info['gaussian_params']
        
        if not parents:
            params = gaussian_params.get((), {'mu': 0, 'sigma': 1})
        else:
            condition = tuple(evidence.get(parent, None) for parent in parents)
            params = gaussian_params.get(condition, {'mu': 0, 'sigma': 1})
        
        # Generar muestras de la distribución gaussiana
        for _ in range(num_samples):
            sample_value = random.gauss(params['mu'], params['sigma'])
            new_evidence = evidence.copy()
            new_evidence[continuous_var] = sample_value
            
            prob = self.get_probability(continuous_var, sample_value, new_evidence)
            total += prob * self.enumerate_all(rest_vars, new_evidence)
        
        return total / num_samples
    
    def query(self, query_var, query_value, evidence=None):
        if evidence is None:
            evidence = {}
        
        all_vars = list(self.nodes.keys())
        
        extended_evidence = evidence.copy()
        extended_evidence[query_var] = query_value
        numerator = self.enumerate_all(all_vars, extended_evidence)
        
        denominator = self.enumerate_all(all_vars, evidence)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def query_continuous_threshold(self, query_var, query_value, evidence=None, threshold_var=None, threshold_value=None):
        """Query especial para variables continuas con umbral"""
        if evidence is None:
            evidence = {}
        
        if threshold_var and threshold_value is not None:
            evidence = evidence.copy()
            evidence[threshold_var] = threshold_value
        
        # Para nodos discretos, usar query normal
        if self.nodes[query_var]['type'] == 'discrete':
            return self.query(query_var, query_value, evidence)
        
        # Para nodos continuos, calcular probabilidad directamente
        return self.get_continuous_probability(query_var, query_value, evidence)
    
    def marginal_probability(self, var, value):
        all_vars = list(self.nodes.keys())
        evidence = {var: value}
        return self.enumerate_all(all_vars, evidence)
    
    def joint_probability(self, assignment):
        prob = 1.0
        for node in self.nodes:
            if node in assignment:
                prob *= self.get_probability(node, assignment[node], assignment)
        return prob
    
    def print_network(self):
        """Imprimir información de la red"""
        print("=== RED BAYESIANA ===")
        print(f"Nodos: {len(self.nodes)}")
        for node, info in self.nodes.items():
            print(f"\n{node}:")
            print(f"  Tipo: {info['type']}")
            print(f"  Estados: {info['states']}")
            print(f"  Padres: {info['parents']}")
            print(f"  Hijos: {info['children']}")
            if info['type'] == 'discrete':
                print(f"  CPT: {info['cpt']}")
            else:
                print(f"  Parámetros Gaussianos: {info['gaussian_params']}")


# def ejemplo_diagnostico_con_temperatura():
#     """Ejemplo: Red bayesiana para diagnóstico médico con temperatura continua"""
    
#     bn = BayesianNetwork()
    
#     # Agregar nodos discretos
#     bn.add_node('Gripe', ['Si', 'No'])
#     bn.add_node('Fiebre', ['Si', 'No'])
#     bn.add_node('Dolor_Cabeza', ['Si', 'No'])
#     bn.add_node('Congestion', ['Si', 'No'])
    
#     # Agregar nodo continuo para temperatura
#     bn.add_continuous_node('Temperatura', parents=['Gripe'])
    
#     # Agregar dependencias
#     bn.add_edge('Gripe', 'Fiebre')
#     bn.add_edge('Gripe', 'Dolor_Cabeza')
#     bn.add_edge('Gripe', 'Congestion')
#     # Temperatura ya tiene Gripe como padre
    
#     # Establecer probabilidades discretas
#     bn.set_cpt('Gripe', {
#         (): {'Si': 0.1, 'No': 0.9}
#     })
    
#     bn.set_cpt('Fiebre', {
#         ('Si',): {'Si': 0.8, 'No': 0.2},
#         ('No',): {'Si': 0.05, 'No': 0.95}
#     })
    
#     bn.set_cpt('Dolor_Cabeza', {
#         ('Si',): {'Si': 0.7, 'No': 0.3},
#         ('No',): {'Si': 0.1, 'No': 0.9}
#     })
    
#     bn.set_cpt('Congestion', {
#         ('Si',): {'Si': 0.9, 'No': 0.1},
#         ('No',): {'Si': 0.02, 'No': 0.98}
#     })
    
#     # Establecer parámetros gaussianos para temperatura
#     # Temperatura normal: ~36.5°C, con gripe: ~38.5°C
#     bn.set_gaussian_params('Temperatura', {
#         ('Si',): {'mu': 38.5, 'sigma': 1.0},  # Con gripe
#         ('No',): {'mu': 36.5, 'sigma': 0.5}   # Sin gripe
#     })
    
#     return bn


# def crear_dataset_con_temperatura():
#     """Crear dataset sintético que incluye temperatura"""
#     dataset = [
#         ({'Fiebre': 'Si', 'Dolor_Cabeza': 'Si', 'Congestion': 'Si', 'Temperatura': 39.2}, 'Si'),
#         ({'Fiebre': 'Si', 'Dolor_Cabeza': 'No', 'Congestion': 'Si', 'Temperatura': 38.8}, 'Si'),
#         ({'Fiebre': 'No', 'Dolor_Cabeza': 'Si', 'Congestion': 'Si', 'Temperatura': 37.1}, 'Si'),
#         ({'Fiebre': 'No', 'Dolor_Cabeza': 'No', 'Congestion': 'Si', 'Temperatura': 36.8}, 'No'),
#         ({'Fiebre': 'Si', 'Dolor_Cabeza': 'Si', 'Congestion': 'No', 'Temperatura': 38.5}, 'Si'),
#         ({'Fiebre': 'No', 'Dolor_Cabeza': 'Si', 'Congestion': 'No', 'Temperatura': 36.2}, 'No'),
#         ({'Fiebre': 'No', 'Dolor_Cabeza': 'No', 'Congestion': 'No', 'Temperatura': 36.4}, 'No'),
#         ({'Fiebre': 'Si', 'Dolor_Cabeza': 'No', 'Congestion': 'No', 'Temperatura': 37.8}, 'Si'),
#         ({'Fiebre': 'No', 'Dolor_Cabeza': 'Si', 'Congestion': 'No', 'Temperatura': 36.1}, 'No'),
#         ({'Fiebre': 'Si', 'Dolor_Cabeza': 'Si', 'Congestion': 'Si', 'Temperatura': 39.5}, 'Si'),
#         # Casos específicos con temperatura ~30°C (hipotermia)
#         ({'Fiebre': 'No', 'Dolor_Cabeza': 'Si', 'Congestion': 'No', 'Temperatura': 30.0}, 'No'),
#         ({'Fiebre': 'No', 'Dolor_Cabeza': 'No', 'Congestion': 'No', 'Temperatura': 29.8}, 'No'),
#     ]
#     return dataset


# def predecir_con_umbral_y_temperatura(red, evidencia, var_objetivo, umbral=0.6):
#     """Predicción considerando también la temperatura"""
#     p_si = red.query(var_objetivo, 'Si', evidencia)
#     if p_si > umbral:
#         return 'Si', p_si
#     else:
#         return 'No', p_si


# def evaluar_accuracy_con_temperatura(red, dataset, var_objetivo, umbral=0.6):
#     """Evaluar accuracy considerando temperatura"""
#     total = len(dataset)
#     correctos = 0
    
#     for evidencia, clase_real in dataset:
#         prediccion, prob = predecir_con_umbral_y_temperatura(red, evidencia, var_objetivo, umbral)
#         if prediccion == clase_real:
#             correctos += 1
        
#         # Mostrar detalles de la predicción
#         temp = evidencia.get('Temperatura', 'N/A')
#         print(f"Temp: {temp}°C, P(Gripe=Si): {prob:.4f}, Pred: {prediccion}, Real: {clase_real}, {'✓' if prediccion == clase_real else '✗'}")
    
#     return correctos / total if total > 0 else 0


# def caso_especial_temperatura_30():
#     """Analizar el caso específico de temperatura = 30°C"""
#     red = ejemplo_diagnostico_con_temperatura()
    
#     # Evidencia: solo temperatura = 30°C
#     evidencia_temp_30 = {'Temperatura': 30.0}
    
#     # Calcular probabilidades
#     p_gripe_si = red.query('Gripe', 'Si', evidencia_temp_30)
#     p_gripe_no = red.query('Gripe', 'No', evidencia_temp_30)
    
#     print("\n=== ANÁLISIS ESPECIAL: TEMPERATURA = 30°C ===")
#     print(f"P(Gripe = Si | Temperatura = 30°C) = {p_gripe_si:.6f}")
#     print(f"P(Gripe = No | Temperatura = 30°C) = {p_gripe_no:.6f}")
    
#     # Análisis: temperatura de 30°C es muy baja (hipotermia)
#     # Según nuestro modelo:
#     # - Con gripe: mu=38.5, sigma=1.0
#     # - Sin gripe: mu=36.5, sigma=0.5
    
#     # Calcular densidades directamente
#     temp_30 = 30.0
#     density_con_gripe = red.gaussian_pdf(temp_30, 38.5, 1.0)
#     density_sin_gripe = red.gaussian_pdf(temp_30, 36.5, 0.5)
    
#     print(f"\nDensidades de probabilidad para T=30°C:")
#     print(f"f(30|Gripe=Si) = {density_con_gripe:.8f}")
#     print(f"f(30|Gripe=No) = {density_sin_gripe:.8f}")
    
   
#     if density_sin_gripe > density_con_gripe:
#         print("\n➤ Conclusión: Es más probable NO tener gripe con T=30°C")
#         print("  (La hipotermia severa es inconsistente con fiebre por gripe)")
#     else:
#         print("\n➤ Conclusión: Es más probable tener gripe con T=30°C")
    
#     return p_gripe_si, p_gripe_no


# if __name__ == "__main__":
#     print("=== RED BAYESIANA CON TEMPERATURA CONTINUA ===\n")
    
 
#     red_medica = ejemplo_diagnostico_con_temperatura()
#     dataset = crear_dataset_con_temperatura()
    
    
#     red_medica.print_network()
    
#     print(f"\n=== EVALUACIÓN CON DATASET ({len(dataset)} casos) ===")
    
    
#     print("\nEvaluando con diferentes umbrales...")
#     for umbral in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
#         print(f"\n--- Umbral = {umbral:.1f} ---")
#         acc = evaluar_accuracy_con_temperatura(red_medica, dataset, 'Gripe', umbral)
#         print(f"Accuracy = {acc:.4f} ({acc*100:.1f}%)")
    
#     # Caso especial: temperatura = 30°C
#     caso_especial_temperatura_30()
    
#     print("\n=== PREDICCIONES ADICIONALES ===")
    
#     # Casos de prueba adicionales
#     casos_prueba = [
#         {'Temperatura': 38.0, 'Fiebre': 'Si'},
#         {'Temperatura': 30.0},
#         {'Temperatura': 39.5, 'Congestion': 'Si'},
#         {'Temperatura': 36.0, 'Fiebre': 'No'}
#     ]
    
#     for i, evidencia in enumerate(casos_prueba, 1):
#         p_si = red_medica.query('Gripe', 'Si', evidencia)
#         print(f"Caso {i}: {evidencia} → P(Gripe=Si) = {p_si:.4f}")