import requests
import pandas as pd
from datetime import datetime, timedelta

# === CONFIGURACIÓN ===
api_key = 'WKTEXQ592KRHZXEFQVXGHU9MU'
location = "Ciudad de Guatemala"
start_date = datetime.strptime("2025-02-01", "%Y-%m-%d")
end_date = datetime.strptime("2025-03-30", "%Y-%m-%d")

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

# === FUNCIÓN PARA PROCESAR UN RANGO DE FECHAS ===
def procesar_rango(inicio, fin):
    print(f"Consultando: {inicio.date()} a {fin.date()}")
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{location}/{inicio.date()}/{fin.date()}"
        f"?unitGroup=metric&include=hours&key={api_key}&contentType=json"
    )
    response = requests.get(url)

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return []

    try:
        data = response.json()
    except Exception as e:
        print("Error al parsear JSON:", e)
        return []

    registros = []
    for dia in data.get('days', []):
        for hora in dia['hours']:
            dt = datetime.strptime(hora['datetime'], "%H:%M:%S")
            fecha_completa = dia['datetime']
            año, mes, dia_ = fecha_completa.split("-")
            hora_num = dt.hour
            franja = clasificar_franja_horaria(hora_num)

            registros.append({
                'target': 1 if hora.get('preciptype') == ['rain'] else 0,
                'year': int(año),
                'mes': int(mes),
                'dia': int(dia_),
                'franja_horaria': franja,
                'temperatura': hora.get('temp'),
                'humedad': hora.get('humidity'),
                'viento_vel_m_s': round(hora.get('windspeed', 0) / 3.6, 2),
                'viento_dir': hora.get('winddir'),
                'presion': hora.get('pressure'),
                'precipitacion': hora.get('precip'),
                'nubosidad': hora.get('cloudcover')
            })
    return registros

# === DIVIDIR EN SEMANAS ===
delta = timedelta(days=7)
actual = start_date
todos_los_datos = []

while actual <= end_date:
    fin_rango = min(actual + delta, end_date)
    datos = procesar_rango(actual, fin_rango)
    todos_los_datos.extend(datos)
    actual = fin_rango + timedelta(days=1)

# === GUARDAR CSV ===
df = pd.DataFrame(todos_los_datos)
df.to_csv("clima_guatemala.csv", index=False)
print(df.head())
