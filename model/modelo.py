from BayesianNetwork import *
import pandas as pd
import pickle
import io

# Variables globales para métricas
TP = 0  # Verdaderos positivos
TN = 0  # Verdaderos negativos
FP = 0  # Falsos positivos
FN = 0  # Falsos negativos

def apriori_Data():
    """Calcula probabilidades a priori de lluvia"""
    df_apriori = pd.read_csv("../data/balanceado_train.csv")
    distribucion = df_apriori['target'].value_counts()
    total = distribucion[0] + distribucion[1]
    p_llueve = distribucion[1] / total
    p_no_llueve = distribucion[0] / total
    
    return p_llueve, p_no_llueve

def get_distribucion_vars():
    """Obtiene las distribuciones de las variables para entrenar la red"""
    df_variables = pd.read_csv("../data/balanceado_train.csv")
    grupos = df_variables.groupby('target')
    stats_continuas = {}
    stats_discretas = {}
    
    for col in df_variables.columns:
        if col == 'target':
            continue
        
        print(f"\nVariable: {col}")
        if pd.api.types.is_numeric_dtype(df_variables[col]):
            # Variables continuas - calcular media y desviación estándar
            medias = grupos[col].mean()
            stds = grupos[col].std()
            
            print("Media por clase target:")
            print(medias)
            print("Desviación estándar (sigma) por clase target:")
            print(stds)
            stats_continuas[col] = {
                'mu': medias.to_dict(),
                'sigma': stds.to_dict()
            }
        else:
            # Variables discretas - calcular probabilidades
            print("Conteo de valores por clase target:")
            conteos = df_variables.groupby(['target', col]).size()
            print(conteos)
            
            # Calcular probabilidades
            probabilidades = conteos / conteos.groupby(level=0).sum()
            print(probabilidades)
            
            stats_discretas[col] = probabilidades
    
    print("Probabilidades discretas: ", stats_discretas)
    return stats_continuas, stats_discretas

def crear_modelo():
    """Crea y entrena la Red Bayesiana"""
    print("Creando modelo de Red Bayesiana...")
    
    # Obtener estadísticas y probabilidades a priori
    stats, stats_discretas = get_distribucion_vars()
    p_llueve, p_no_llueve = apriori_Data()
    
    # Crear la red bayesiana
    bn = BayesianNetwork()
    
    # Agregar nodos
    bn.add_node('Llueve', ['Si', 'No'])
    bn.add_node('Franja_Horaria', ['Madrugada', 'Tarde', 'Morning', 'Noche'])
    bn.add_continuous_node('Temperatura', parents=['Llueve'])
    bn.add_continuous_node('Humedad', parents=['Llueve'])
    bn.add_continuous_node('Velocidad_Viento', parents=['Llueve'])
    bn.add_continuous_node('Viento_Direccion', parents=['Llueve'])
    bn.add_continuous_node('Presion', parents=['Llueve'])
    bn.add_continuous_node('Nubosidad', parents=['Llueve'])
    
    # Agregar aristas
    bn.add_edge('Llueve', 'Franja_Horaria')
    
    # Configurar CPT para nodo raíz (Llueve)
    bn.set_cpt('Llueve', {
        (): {'Si': p_llueve, 'No': p_no_llueve}
    })
    
    # Configurar CPT para variable discreta (Franja_Horaria)
    bn.set_cpt('Franja_Horaria', {
        ('Si',): {
            'Madrugada': stats_discretas['franja_horaria'][(1, 'Madrugada')],
            'Morning': stats_discretas['franja_horaria'][(1, 'Morning')],
            'Tarde': stats_discretas['franja_horaria'][(1, 'Tarde')],
            'Noche': stats_discretas['franja_horaria'][(1, 'Noche')]
        },
        ('No',): {
            'Madrugada': stats_discretas['franja_horaria'][(0, 'Madrugada')],
            'Morning': stats_discretas['franja_horaria'][(0, 'Morning')],
            'Tarde': stats_discretas['franja_horaria'][(0, 'Tarde')],
            'Noche': stats_discretas['franja_horaria'][(0, 'Noche')]
        }
    })
    
    # Configurar parámetros gaussianos para variables continuas
    variables_continuas = ['Temperatura', 'Humedad', 'Velocidad_Viento', 
                          'Viento_Direccion', 'Presion', 'Nubosidad']
    mapeo_nombres = {
        'Temperatura': 'temperatura',
        'Humedad': 'humedad',
        'Velocidad_Viento': 'viento_vel_m_s',
        'Viento_Direccion': 'viento_dir',
        'Presion': 'presion',
        'Nubosidad': 'nubosidad'
    }
    
    for var in variables_continuas:
        nombre_data = mapeo_nombres[var]
        bn.set_gaussian_params(var, {
            ('Si',): {
                'mu': stats[nombre_data]['mu'][1],
                'sigma': stats[nombre_data]['sigma'][1]
            },
            ('No',): {
                'mu': stats[nombre_data]['mu'][0],
                'sigma': stats[nombre_data]['sigma'][0]
            }
        })
    
    print("Modelo creado exitosamente!")
    bn.print_network()
    return bn

def guardar_modelo(modelo, nombre_archivo='modelo_lluvia.pkl'):
    """Guarda el modelo usando pickle"""
    try:
        with open(nombre_archivo, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo guardado exitosamente como {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

def cargar_modelo(nombre_archivo='modelo_lluvia.pkl'):
    """Carga el modelo desde un archivo pickle"""
    try:
        with open(nombre_archivo, 'rb') as f:
            modelo = pickle.load(f)
        print(f"Modelo cargado exitosamente desde {nombre_archivo}")
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def predecir_lluvia(modelo, datos_clima):
    """
    Predice si va a llover basado en los datos climáticos
    
    Args:
        modelo: Red Bayesiana entrenada
        datos_clima: dict con las variables climáticas
                    {
                        'temperatura': float,
                        'humedad': float,
                        'viento_vel_m_s': float,
                        'viento_dir': float,
                        'presion': float,
                        'nubosidad': float,
                        'franja_horaria': str ('Madrugada', 'Morning', 'Tarde', 'Noche')
                    }
    
    Returns:
        dict: {
            'probabilidad_lluvia': float,
            'prediccion': str ('Sí va a llover' o 'No va a llover'),
            'confianza': str
        }
    """
    try:
        # Mapear nombres de variables para la consulta
        evidencia = {
            'Temperatura': datos_clima['temperatura'],
            'Humedad': datos_clima['humedad'],
            'Velocidad_Viento': datos_clima['viento_vel_m_s'],
            'Viento_Direccion': datos_clima['viento_dir'],
            'Presion': datos_clima['presion'],
            'Nubosidad': datos_clima['nubosidad'],
            'Franja_Horaria': datos_clima['franja_horaria']
        }
        
        # Calcular probabilidad de lluvia
        prob_lluvia = modelo.query('Llueve', 'Si', evidencia)
        
        # Determinar predicción (umbral de 0.55)
        umbral = 0.55
        prediccion = "Sí va a llover" if prob_lluvia > umbral else "No va a llover"
        
        # Determinar nivel de confianza
        if prob_lluvia > 0.8 or prob_lluvia < 0.2:
            confianza = "Alta"
        elif prob_lluvia > 0.65 or prob_lluvia < 0.35:
            confianza = "Media"
        else:
            confianza = "Baja"
        
        return {
            'probabilidad_lluvia': round(prob_lluvia, 4),
            'prediccion': prediccion,
            'confianza': confianza
        }
        
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return None

def evaluar_modelo(modelo, archivo_test="../data/marzo-abril.csv"):
    """Evalúa el modelo con datos de prueba"""
    global TP, TN, FP, FN
    TP = TN = FP = FN = 0
    
    df = pd.read_csv(archivo_test)
    
    print("Evaluando modelo...")
    for idx, row in df.iterrows():
        evidencia = {
            'Temperatura': row['temperatura'],
            'Humedad': row['humedad'],
            'Velocidad_Viento': row['viento_vel_m_s'],
            'Viento_Direccion': row['viento_dir'],
            'Presion': row['presion'],
            'Franja_Horaria': row['franja_horaria'],
            'Nubosidad': row['nubosidad']
        }
        
        p = modelo.query('Llueve', 'Si', evidencia)
        prediccion = 1 if p > 0.55 else 0
        real = row['target']
        
        if prediccion == 1 and real == 1:
            TP += 1
        elif prediccion == 0 and real == 0:
            TN += 1
        elif prediccion == 1 and real == 0:
            FP += 1
        elif prediccion == 0 and real == 1:
            FN += 1
    
    # Calcular métricas
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== MÉTRICAS DEL MODELO ===")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    
    print(f"\n=== MATRIZ DE CONFUSIÓN ===")
    print(f"               Predicción Positiva   Predicción Negativa")
    print(f"Real Positivo      {TP}                    {FN}")
    print(f"Real Negativo      {FP}                    {TN}")

def ejemplo_uso():
    """Ejemplo de cómo usar el modelo para predicciones"""
    print("\n=== EJEMPLO DE USO ===")
    
    # Cargar modelo guardado
    modelo = cargar_modelo()
    
    if modelo is None:
        print("No se pudo cargar el modelo. Creando uno nuevo...")
        modelo = crear_modelo()
        guardar_modelo(modelo)
    
    # Ejemplo de predicción
    datos_ejemplo = {
        'temperatura': 24.0,
        'humedad': 47.01,
        'viento_vel_m_s': 5.11,
        'viento_dir': 20.0,
        'presion': 1013,
        'nubosidad': 30.0,
        'franja_horaria': 'Noche'
    }
    
    resultado = predecir_lluvia(modelo, datos_ejemplo)
    
    if resultado:
        print(f"\n=== PREDICCIÓN ===")
        print(f"Datos de entrada: {datos_ejemplo}")
        print(f"Probabilidad de lluvia: {resultado['probabilidad_lluvia']:.2%}")
        print(f"Predicción: {resultado['prediccion']}")
        print(f"Confianza: {resultado['confianza']}")

if __name__ == "__main__":
    # Crear y entrenar el modelo
    modelo = crear_modelo()
    
    # Guardar el modelo
    guardar_modelo(modelo)
    
    # Evaluar el modelo
    evaluar_modelo(modelo)
    
    # Mostrar ejemplo de uso
    ejemplo_uso()