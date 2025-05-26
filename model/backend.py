from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import requests
import pickle
from datetime import datetime

app = FastAPI()

# === CONFIG ===
API_KEY = 'WKTEXQ592KRHZXEFQVXGHU9MU'
LOCATION = 'Ciudad de Guatemala'
MODEL_PATH = 'modelo_lluvia.pkl'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://riskrain.netlify.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Utility Functions ===
def clasificar_franja_horaria(hora: int) -> str:
    if 0 <= hora < 6:
        return 'Madrugada'
    elif 6 <= hora < 12:
        return 'Morning'
    elif 12 <= hora < 18:
        return 'Tarde'
    else:
        return 'Noche'

def cargar_modelo(nombre_archivo='modelo_lluvia.pkl'):
    try:
        with open(nombre_archivo, 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def predecir_lluvia(modelo, datos_clima: dict):
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

    umbral = 0.55
    prediccion = "Sí va a llover" if prob_lluvia > umbral else "No va a llover"

    if prob_lluvia > 0.8 or prob_lluvia < 0.2:
        confianza = "Alta"
    elif prob_lluvia > 0.65 or prob_lluvia < 0.35:
        confianza = "Media"
    else:
        confianza = "Baja"

    return {
        'probabilidad_lluvia': round(prob_lluvia, 4),
        'prediccion': prediccion,
        'confianza': confianza,
    }

# === API Route ===
@app.get("/predict")
def predict():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{LOCATION}/{today}/{today}"
        f"?unitGroup=metric&include=hours&key={API_KEY}&contentType=json"
    )

    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        current_hour = now.hour

        # Find the matching hour entry
        hour_data = next((h for d in data['days'] for h in d['hours'] if int(h['datetime'].split(":")[0]) == current_hour), None)

        if not hour_data:
            return {"error": "No se encontró información horaria para este momento."}

        franja = clasificar_franja_horaria(current_hour)
        datos_clima = {
            'temperatura': hour_data.get('temp'),
            'humedad': hour_data.get('humidity'),
            'viento_vel_m_s': round(hour_data.get('windspeed', 0) / 3.6, 2),
            'viento_dir': hour_data.get('winddir'),
            'presion': hour_data.get('pressure'),
            'nubosidad': hour_data.get('cloudcover'),
            'franja_horaria': franja
        }

        modelo = cargar_modelo(MODEL_PATH)
        if not modelo:
            return {"error": "No se pudo cargar el modelo."}

        resultado = predecir_lluvia(modelo, datos_clima)
        return {
            **resultado,
            "evidencia": datos_clima
        }


    except Exception as e:
        return {"error": f"Fallo en la predicción: {str(e)}"}
