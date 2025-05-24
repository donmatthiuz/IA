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
            'cpt': {}
        }
        self.edges[name] = []
    
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
    
    def get_probability(self, node, value, evidence=None):
        
        if evidence is None:
            evidence = {}
        
        cpt = self.nodes[node]['cpt']
        parents = self.nodes[node]['parents']
        
        if not parents:
      
            return cpt.get((), {}).get(value, 0)
        else:
          
            condition = tuple(evidence.get(parent, None) for parent in parents)
            return cpt.get(condition, {}).get(value, 0)
    
    def enumerate_all(self, vars_list, evidence):
        
        if not vars_list:
            return 1.0
        
        first_var = vars_list[0]
        rest_vars = vars_list[1:]
        
        if first_var in evidence:
         
            prob = self.get_probability(first_var, evidence[first_var], evidence)
            return prob * self.enumerate_all(rest_vars, evidence)
        else:
          
            total = 0.0
            for value in self.nodes[first_var]['states']:
                new_evidence = evidence.copy()
                new_evidence[first_var] = value
                prob = self.get_probability(first_var, value, new_evidence)
                total += prob * self.enumerate_all(rest_vars, new_evidence)
            return total
    
    def query(self, query_var, query_value, evidence=None):
        if evidence is None:
            evidence = {}
        
        
        all_vars = list(self.nodes.keys())
        
       
        extended_evidence = evidence.copy()
        extended_evidence[query_var] = query_value
        numerator = self.enumerate_all(all_vars, extended_evidence)
        
        # Calcular P(evidence)
        denominator = self.enumerate_all(all_vars, evidence)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
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
            print(f"  Estados: {info['states']}")
            print(f"  Padres: {info['parents']}")
            print(f"  Hijos: {info['children']}")
            print(f"  CPT: {info['cpt']}")


# Ejemplo de uso: Red para diagnóstico médico
def ejemplo_diagnostico_medico():
    """Ejemplo: Red bayesiana para diagnóstico médico"""
    
    # Crear la red
    bn = BayesianNetwork()
    
    # Agregar nodos
    bn.add_node('Gripe', ['Si', 'No'])
    bn.add_node('Fiebre', ['Si', 'No'])
    bn.add_node('Dolor_Cabeza', ['Si', 'No'])
    bn.add_node('Congestion', ['Si', 'No'])
    
    # Agregar dependencias
    bn.add_edge('Gripe', 'Fiebre')
    bn.add_edge('Gripe', 'Dolor_Cabeza')
    bn.add_edge('Gripe', 'Congestion')
    
    # Establecer probabilidades
    # P(Gripe) - probabilidad a priori
    bn.set_cpt('Gripe', {
        (): {'Si': 0.1, 'No': 0.9}
    })
    
    # P(Fiebre | Gripe)
    bn.set_cpt('Fiebre', {
        ('Si',): {'Si': 0.8, 'No': 0.2},  # Si tiene gripe
        ('No',): {'Si': 0.05, 'No': 0.95}  # Si no tiene gripe
    })
    
    # P(Dolor_Cabeza | Gripe)
    bn.set_cpt('Dolor_Cabeza', {
        ('Si',): {'Si': 0.7, 'No': 0.3},  # Si tiene gripe
        ('No',): {'Si': 0.1, 'No': 0.9}   # Si no tiene gripe
    })
    
    # P(Congestion | Gripe)
    bn.set_cpt('Congestion', {
        ('Si',): {'Si': 0.9, 'No': 0.1},  # Si tiene gripe
        ('No',): {'Si': 0.02, 'No': 0.98} # Si no tiene gripe
    })
    
    return bn


def ejemplo_clima():
    """Ejemplo: Red bayesiana para predicción del clima"""
    
    bn = BayesianNetwork()
    
    # Nodos
    bn.add_node('Nublado', ['Si', 'No'])
    bn.add_node('Lluvia', ['Si', 'No'])
    bn.add_node('Aspersores', ['Si', 'No'])
    bn.add_node('Cesped_Mojado', ['Si', 'No'])
    
    # Dependencias
    bn.add_edge('Nublado', 'Lluvia')
    bn.add_edge('Lluvia', 'Cesped_Mojado')
    bn.add_edge('Aspersores', 'Cesped_Mojado')
    
    # Probabilidades
    bn.set_cpt('Nublado', {
        (): {'Si': 0.5, 'No': 0.5}
    })
    
    bn.set_cpt('Aspersores', {
        (): {'Si': 0.1, 'No': 0.9}
    })
    
    bn.set_cpt('Lluvia', {
        ('Si',): {'Si': 0.8, 'No': 0.2},  # Si está nublado
        ('No',): {'Si': 0.2, 'No': 0.8}   # Si no está nublado
    })
    
    bn.set_cpt('Cesped_Mojado', {
        ('Si', 'Si'): {'Si': 0.99, 'No': 0.01},  # Lluvia=Si, Aspersores=Si
        ('Si', 'No'): {'Si': 0.9, 'No': 0.1},    # Lluvia=Si, Aspersores=No
        ('No', 'Si'): {'Si': 0.9, 'No': 0.1},    # Lluvia=No, Aspersores=Si
        ('No', 'No'): {'Si': 0.01, 'No': 0.99}   # Lluvia=No, Aspersores=No
    })
    
    return bn

def crear_dataset_sintetico():
    # Generar combinaciones simples de síntomas y su clase real
    dataset = [
        ({'Fiebre': 'Si', 'Dolor_Cabeza': 'Si', 'Congestion': 'Si'}, 'Si'),
        ({'Fiebre': 'Si', 'Dolor_Cabeza': 'No', 'Congestion': 'Si'}, 'Si'),
        ({'Fiebre': 'No', 'Dolor_Cabeza': 'Si', 'Congestion': 'Si'}, 'Si'),
        ({'Fiebre': 'No', 'Dolor_Cabeza': 'No', 'Congestion': 'Si'}, 'No'),
        ({'Fiebre': 'Si', 'Dolor_Cabeza': 'Si', 'Congestion': 'No'}, 'Si'),
        ({'Fiebre': 'No', 'Dolor_Cabeza': 'Si', 'Congestion': 'No'}, 'No'),
        ({'Fiebre': 'No', 'Dolor_Cabeza': 'No', 'Congestion': 'No'}, 'No'),
        ({'Fiebre': 'Si', 'Dolor_Cabeza': 'No', 'Congestion': 'No'}, 'Si'),
        ({'Fiebre': 'No', 'Dolor_Cabeza': 'Si', 'Congestion': 'No'}, 'No'),
        ({'Fiebre': 'Si', 'Dolor_Cabeza': 'Si', 'Congestion': 'Si'}, 'Si'),
    ]
    return dataset

def predecir_con_umbral(red, evidencia, var_objetivo, umbral=0.6):
    # Obtener probabilidad de "Si"
    p_si = red.query(var_objetivo, 'Si', evidencia)
    if p_si > umbral:
        return 'Si', p_si
    else:
        return 'No', p_si

def evaluar_accuracy(red, dataset, var_objetivo, umbral=0.6):
    total = len(dataset)
    correctos = 0
    for evidencia, clase_real in dataset:
        prediccion, prob = predecir_con_umbral(red, evidencia, var_objetivo, umbral)
        if prediccion == clase_real:
            correctos += 1
    return correctos / total if total > 0 else 0


if __name__ == "__main__":
    red_medica = ejemplo_diagnostico_medico()
    dataset = crear_dataset_sintetico()

    print("Evaluando con diferentes umbrales de probabilidad para 'Si'...")

    for umbral in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        acc = evaluar_accuracy(red_medica, dataset, 'Gripe', umbral)
        print(f"Umbral={umbral:.1f} => Accuracy={acc:.4f}")
    