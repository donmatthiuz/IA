from continuos import *
import pandas as pd
import itertools
from collections import defaultdict
import time

def apriori_Data():
    df_apriori = pd.read_csv("../data/balanceado_train.csv")
    distribucion = df_apriori['target'].value_counts()
    total = distribucion[0] + distribucion[1]
    p_llueve = distribucion[1] / total
    p_no_llueve = distribucion[0] / total
    return p_llueve, p_no_llueve

def get_distribucion_vars():
    df_variables = pd.read_csv("../data/balanceado_train.csv")
    grupos = df_variables.groupby('target')
    stats_continuas = {} 

    for col in df_variables.columns:
        if col == 'target':
            continue

        if pd.api.types.is_numeric_dtype(df_variables[col]):
            medias = grupos[col].mean()
            stds = grupos[col].std()
            stats_continuas[col] = {
                'mu': medias.to_dict(),
                'sigma': stds.to_dict()
            }
    
    return stats_continuas

def get_conditional_stats(df, target_var, parent_vars, condition_values):
    """
    Obtiene estadísticas condicionales para una variable dada sus padres
    """
    # Crear máscara para filtrar por condiciones de los padres
    mask = pd.Series([True] * len(df))
    
    for parent, value in zip(parent_vars, condition_values):
        if parent == 'target':  # Llueve
            mask = mask & (df['target'] == value)
        else:
           
            mask = mask & (df[parent] == value)
    
    filtered_data = df[mask][target_var]
    
    if len(filtered_data) > 0:
        return {
            'mu': filtered_data.mean(),
            'sigma': filtered_data.std() if filtered_data.std() > 0 else 0.01  # Evitar sigma = 0
        }
    else:
        # Si no hay datos, usar valores por defecto
        return {'mu': df[target_var].mean(), 'sigma': df[target_var].std()}

def evaluate_network_configuration(config, df, threshold=0.55):
    """
    Evalúa una configuración específica de la red bayesiana
    """
    try:
        p_llueve, p_no_llueve = apriori_Data()
        stats = get_distribucion_vars()
        
        # Crear la red bayesiana
        bn = BayesianNetwork()
        bn.add_node('Llueve', ['Si', 'No'])
        
        # Agregar nodos con sus padres según la configuración
        for node, parents in config.items():
            bn.add_continuous_node(node, parents=parents)
        
        # Configurar probabilidades a priori para Llueve
        bn.set_cpt('Llueve', {
            (): {'Si': p_llueve, 'No': p_no_llueve}
        })
        
        # Configurar parámetros gaussianos para cada nodo
        variable_mapping = {
            'Temperatura': 'temperatura',
            'Humedad': 'humedad',
            'Velocidad_Viento': 'viento_vel_m_s',
            'Viento_Direccion': 'viento_dir',
            'Presion': 'presion',
            'Nubosidad': 'nubosidad'
        }
        
        for node, parents in config.items():
            var_name = variable_mapping[node]
            
            if parents == ['Llueve']:
                # Caso simple: solo depende de Llueve
                bn.set_gaussian_params(node, {
                    ('Si',): {'mu': stats[var_name]['mu'][1], 
                             'sigma': stats[var_name]['sigma'][1]}, 
                    ('No',): {'mu': stats[var_name]['mu'][0], 
                             'sigma': stats[var_name]['sigma'][0]}
                })
            else:
                # Caso más complejo: múltiples padres
                # Para simplificar, usaremos solo la dependencia de Llueve
                # En una implementación más compleja, habría que considerar todas las combinaciones
                bn.set_gaussian_params(node, {
                    ('Si',): {'mu': stats[var_name]['mu'][1], 
                             'sigma': stats[var_name]['sigma'][1]}, 
                    ('No',): {'mu': stats[var_name]['mu'][0], 
                             'sigma': stats[var_name]['sigma'][0]}
                })
        
        # Evaluar el modelo
        TP = TN = FP = FN = 0
        
        for idx, row in df.iterrows():
            evidencia = {}
            for node in config.keys():
                var_name = variable_mapping[node]
                evidencia[node] = row[var_name]
            
            p = bn.query('Llueve', 'Si', evidencia)
            prediccion = 1 if p > threshold else 0
            real = row['target']

            if prediccion == 1 and real == 1:
                TP += 1
            elif prediccion == 0 and real == 0:
                TN += 1
            elif prediccion == 1 and real == 0:
                FP += 1
            elif prediccion == 0 and real == 1:
                FN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }
    
    except Exception as e:
        print(f"Error evaluating configuration: {e}")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

def generate_configurations():
    """
    Genera diferentes configuraciones de padres para probar
    """
    nodes = ['Temperatura', 'Humedad', 'Velocidad_Viento', 'Viento_Direccion', 'Presion', 'Nubosidad']
    
    configurations = []
    
    # Configuración 1: Solo Llueve como padre (original)
    config1 = {node: ['Llueve'] for node in nodes}
    configurations.append(("Solo_Llueve", config1))
    
    # Configuración 2: Temperatura y Humedad se influyen mutuamente
    config2 = {
        'Temperatura': ['Llueve'],
        'Humedad': ['Llueve', 'Temperatura'],
        'Velocidad_Viento': ['Llueve'],
        'Viento_Direccion': ['Llueve'],
        'Presion': ['Llueve'],
        'Nubosidad': ['Llueve']
    }
    configurations.append(("Temp_influye_Humedad", config2))
    
    # Configuración 3: Presión influye en temperatura y humedad
    config3 = {
        'Presion': ['Llueve'],
        'Temperatura': ['Llueve', 'Presion'],
        'Humedad': ['Llueve', 'Presion'],
        'Velocidad_Viento': ['Llueve'],
        'Viento_Direccion': ['Llueve'],
        'Nubosidad': ['Llueve']
    }
    configurations.append(("Presion_influye_Temp_Hum", config3))
    
    # Configuración 4: Viento direccion influye velocidad
    config4 = {
        'Temperatura': ['Llueve'],
        'Humedad': ['Llueve'],
        'Viento_Direccion': ['Llueve'],
        'Velocidad_Viento': ['Llueve', 'Viento_Direccion'],
        'Presion': ['Llueve'],
        'Nubosidad': ['Llueve']
    }
    configurations.append(("Dir_Viento_influye_Vel", config4))
    
    # Configuración 5: Nubosidad influye en temperatura y humedad
    config5 = {
        'Nubosidad': ['Llueve'],
        'Temperatura': ['Llueve', 'Nubosidad'],
        'Humedad': ['Llueve', 'Nubosidad'],
        'Velocidad_Viento': ['Llueve'],
        'Viento_Direccion': ['Llueve'],
        'Presion': ['Llueve']
    }
    configurations.append(("Nubosidad_influye_Temp_Hum", config5))
    
    # Configuración 6: Cadena de dependencias
    config6 = {
        'Presion': ['Llueve'],
        'Temperatura': ['Llueve', 'Presion'],
        'Humedad': ['Llueve', 'Temperatura'],
        'Nubosidad': ['Llueve', 'Humedad'],
        'Velocidad_Viento': ['Llueve'],
        'Viento_Direccion': ['Llueve']
    }
    configurations.append(("Cadena_Presion_Temp_Hum_Nub", config6))
    
    # Configuración 7: Todas las variables meteorológicas se influyen
    config7 = {
        'Presion': ['Llueve'],
        'Temperatura': ['Llueve', 'Presion'],
        'Humedad': ['Llueve', 'Temperatura', 'Presion'],
        'Nubosidad': ['Llueve', 'Humedad'],
        'Velocidad_Viento': ['Llueve', 'Presion'],
        'Viento_Direccion': ['Llueve', 'Velocidad_Viento']
    }
    configurations.append(("Interdependencias_Complejas", config7))
    
    return configurations

def main():
    # Cargar datos
    df = pd.read_csv("../data/balanceado_train.csv")
    
    print("Evaluando diferentes configuraciones de red bayesiana...")
    print("=" * 60)
    
    # Generar configuraciones
    configurations = generate_configurations()
    
    results = []
    
    for name, config in configurations:
        print(f"\nEvaluando configuración: {name}")
        print("-" * 40)
        
        start_time = time.time()
        metrics = evaluate_network_configuration(config, df)
        end_time = time.time()
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        print(f"Tiempo: {end_time - start_time:.2f}s")
        
        results.append({
            'name': name,
            'config': config,
            'metrics': metrics,
            'time': end_time - start_time
        })
    
    # Encontrar la mejor configuración
    best_config = max(results, key=lambda x: x['metrics']['accuracy'])
    
    print("\n" + "=" * 60)
    print("MEJOR CONFIGURACIÓN ENCONTRADA:")
    print("=" * 60)
    print(f"Nombre: {best_config['name']}")
    print(f"Accuracy: {best_config['metrics']['accuracy']:.4f}")
    print(f"Precision: {best_config['metrics']['precision']:.4f}")
    print(f"Recall: {best_config['metrics']['recall']:.4f}")
    print(f"F1-score: {best_config['metrics']['f1']:.4f}")
    print(f"TP: {best_config['metrics']['TP']}")
    print(f"TN: {best_config['metrics']['TN']}")
    print(f"FP: {best_config['metrics']['FP']}")
    print(f"FN: {best_config['metrics']['FN']}")
    
    print("\nConfiguración de padres:")
    for node, parents in best_config['config'].items():
        print(f"  {node}: {parents}")
    
    # Mostrar ranking completo
    print("\n" + "=" * 60)
    print("RANKING COMPLETO (por accuracy):")
    print("=" * 60)
    sorted_results = sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['name']}: {result['metrics']['accuracy']:.4f}")
    
    return best_config, results

if __name__ == "__main__":
    best_config, all_results = main()