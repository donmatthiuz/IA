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

