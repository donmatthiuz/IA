from continuos import *
import pandas as pd

import io

df = pd.read_csv("../data/balanceado_test.csv")

TP = 0  # Verdaderos positivos
TN = 0  # Verdaderos negativos
FP = 0  # Falsos positivos
FN = 0  # Falsos negativos

def apriori_Data ():
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

        print(f"\nVariable: {col}")
        if pd.api.types.is_numeric_dtype(df_variables[col]):
            # Mostrar media por grupo target
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
            # Mostrar conteos de valores por grupo target
            print("Conteo de valores por clase target:")
            conteos = df_variables.groupby(['target', col]).size()
            print(conteos)
    
    return stats_continuas

  


if __name__ == "__main__":
  stats =get_distribucion_vars()

  
  p_llueve, p_no_llueve = apriori_Data()




  bn = BayesianNetwork()
  bn.add_node('Llueve', ['Si', 'No'])
  bn.add_continuous_node('Temperatura', parents=['Llueve'])
  bn.add_continuous_node('Humedad', parents=['Llueve'])
  bn.add_continuous_node('Velocidad_Viento', parents=['Llueve'])
  bn.add_continuous_node('Viento_Direccion', parents=['Llueve'])
  bn.add_continuous_node('Presion', parents=['Llueve'])
  # bn.add_continuous_node('Precipitacion', parents=['Llueve'])
  bn.add_continuous_node('Nubosidad', parents=['Llueve'])

  bn.set_cpt('Llueve', {
         (): {'Si': p_llueve, 'No': p_no_llueve}
  })

  
  bn.set_gaussian_params('Temperatura', {
        ('Si',): {'mu':  stats['temperatura']['mu'][1] , 
                    'sigma': stats['temperatura']['sigma'][1]}, 
        ('No',): {'mu':  stats['temperatura']['mu'][0] , 'sigma': stats['temperatura']['sigma'][0]}
    })
  
  bn.set_gaussian_params('Humedad', {
        ('Si',): {'mu':  stats['humedad']['mu'][1] , 
                    'sigma': stats['humedad']['sigma'][1]}, 
        ('No',): {'mu':  stats['humedad']['mu'][0] , 
                    'sigma': stats['humedad']['sigma'][0]}
    })
  
  
  

  bn.set_gaussian_params('Velocidad_Viento', {
        ('Si',): {'mu':  stats['viento_vel_m_s']['mu'][1] , 
                    'sigma': stats['viento_vel_m_s']['sigma'][1]}, 
        ('No',): {'mu':  stats['viento_vel_m_s']['mu'][0] , 
                    'sigma': stats['viento_vel_m_s']['sigma'][0]}
    })
  
  bn.set_gaussian_params('Viento_Direccion', {
        ('Si',): {'mu':  stats['viento_dir']['mu'][1] , 
                    'sigma': stats['viento_dir']['sigma'][1]}, 
        ('No',): {'mu':  stats['viento_dir']['mu'][0] , 
                    'sigma': stats['viento_dir']['sigma'][0]}
    })
  
  bn.set_gaussian_params('Presion', {
        ('Si',): {'mu':  stats['presion']['mu'][1] , 
                    'sigma': stats['presion']['sigma'][1]}, 
        ('No',): {'mu':  stats['presion']['mu'][0] , 
                    'sigma': stats['presion']['sigma'][0]}
    })
  
  # bn.set_gaussian_params('Precipitacion', {
  #       ('Si',): {'mu':  stats['precipitacion']['mu'][1] , 
  #                   'sigma': stats['precipitacion']['sigma'][1]}, 
  #       ('No',): {'mu':  stats['precipitacion']['mu'][0] , 
  #                   'sigma': stats['precipitacion']['sigma'][0]}
  #   })
  
  bn.set_gaussian_params('Nubosidad', {
        ('Si',): {'mu':  stats['nubosidad']['mu'][1] , 
                    'sigma': stats['nubosidad']['sigma'][1]}, 
        ('No',): {'mu':  stats['nubosidad']['mu'][0] , 
                    'sigma': stats['nubosidad']['sigma'][0]}
    })
  
  bn.print_network()
  
  evidencia = {
    'Temperatura': 24.0,
    'Humedad': 47.01,
    'Velocidad_Viento': 5.11,
    'Viento_Direccion': 20.0,
    'Presion': 1013,
    'Precipitacion': 0.0,
    'Nubosidad': 30.0
  }
  p = bn.query('Llueve', 'Si', evidencia)
  print(p)


  for idx, row in df.iterrows():
    evidencia = {
        'Temperatura': row['temperatura'],
        'Humedad': row['humedad'],
        'Velocidad_Viento': row['viento_vel_m_s'],
        'Viento_Direccion': row['viento_dir'],
        'Presion': row['presion'],
        # 'Precipitacion': row['precipitacion'],
        'Nubosidad': row['nubosidad']
    }
    p = bn.query('Llueve', 'Si', evidencia)
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


accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")