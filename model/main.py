
"""
Ejemplo simple: Carga el modelo y hace una predicción con datos definidos
"""

import pickle

def cargar_modelo(nombre_archivo='modelo_lluvia.pkl'):
    try:
        with open(nombre_archivo, 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def predecir_lluvia(modelo, datos_clima):
    evidencia = {
        'Temperatura': datos_clima['temperatura'],
        'Humedad': datos_clima['humedad'],
        'Velocidad_Viento': datos_clima['viento_vel_m_s'],
        'Viento_Direccion': datos_clima['viento_dir'],
        'Presion': datos_clima['presion'],
        'Nubosidad': datos_clima['nubosidad'],
        'Franja_Horaria': datos_clima['franja_horaria']
    }
    prob_lluvia = modelo.query('Llueve', 'Si', evidencia)
    return prob_lluvia

def main():
    modelo = cargar_modelo()
    if not modelo:
        return

    datos = {
        'temperatura': 21.0,
        'humedad': 77.91,
        'viento_vel_m_s': 5.11,
        'viento_dir': 200.0,
        'presion': 1018.0,
        'nubosidad': 89.0,
        'franja_horaria': 'Noche'
    }

    prob = predecir_lluvia(modelo, datos)
    print(f"Probabilidad de lluvia: {prob:.2%}")

if __name__ == "__main__":
    main()
