import requests
import pandas as pd
from datetime import datetime

# === CONFIGURACIÓN ===
api_key = 'API_KEY' #Tu propia API Key de VisualCrossing
location = "Ciudad de Guatemala"
start_date = "2025-04-15"
end_date = "2025-05-15"

# === FUNCIÓN PARA CLASIFICAR LA FRANJA HORARIA ===
def clasificar_franja_horaria(hora):
    h = int(hora)
    if 0 <= h < 6:
        return 'Madrugada'
    elif 6 <= h < 12:
        return 'Morning'
    elif 12 <= h < 18:
        return 'Tarde'
    else:
        return 'Noche'

# === LLAMADA A LA API ===
url = (
    f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    f"{location}/{start_date}/{end_date}"
    f"?unitGroup=metric&include=hours&key={api_key}&contentType=json"
)

response = requests.get(url)
data = response.json()

# === PROCESAR DATOS POR HORA ===
registros = []
for dia in data['days']:
    for hora in dia['hours']:
        dt = datetime.strptime(hora['datetime'], "%H:%M:%S")
        fecha_completa = dia['datetime']
        año, mes, dia_ = fecha_completa.split("-")
        hora_num = dt.hour
        franja = clasificar_franja_horaria(hora_num)

        registros.append({
            'year': int(año),
            'mes': int(mes),
            'dia': int(dia_),
            'franja_horaria': franja,
            'temperatura': hora.get('temp'),
            'humedad': hora.get('humidity'),
            'viento_vel_m_s': round(hora.get('windspeed', 0) / 3.6, 2),  # de km/h a m/s
            'viento_dir': hora.get('winddir'),
            'presion': hora.get('pressure'),
            'precipitacion': hora.get('precip'),
            'nubosidad': hora.get('cloudcover')
        })

# === GUARDAR A CSV ===
df = pd.DataFrame(registros)
df.to_csv("clima_guatemala.csv", index=False)
print(df.head())