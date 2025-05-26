import { useState, useEffect } from "react"
import { RefreshCw, CloudRain, Sun, Cloud, Loader2 } from 'lucide-react'
import "./App.css"

interface Evidence {
  temperatura: number
  humedad: number
  viento_vel_m_s: number
  viento_dir: number
  presion: number
  nubosidad: number
  franja_horaria: string
}

interface PredictionResponse {
  probabilidad_lluvia: number
  prediccion: string
  confianza: string
  evidencia: Evidence
}

function App() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchPrediction = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })

      if (!response.ok) {
        throw new Error("Failed to get prediction")
      }

      const data: PredictionResponse = await response.json()
      setPrediction(data)
    } catch (err) {
      setError("Error connecting to the prediction service. Please make sure the backend is running.")
      console.error("Prediction error:", err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPrediction()
  }, [])

  const getRainIcon = (probability: number) => {
    if (probability > 0.7) return <CloudRain size={80} className="rain-icon" />
    if (probability > 0.3) return <Cloud size={80} className="cloud-icon" />
    return <Sun size={80} className="sun-icon" />
  }

  const getConfidenceClass = (confianza: string) => {
    switch (confianza.toLowerCase()) {
      case "alta":
        return "confidence-high"
      case "media":
        return "confidence-medium"
      case "baja":
        return "confidence-low"
      default:
        return "confidence-default"
    }
  }

  const getRiskClass = (probability: number) => {
    if (probability > 0.7) return "risk-high"
    if (probability > 0.4) return "risk-medium"
    return "risk-low"
  }

  const formatWindDirection = (degrees: number) => {
    const directions = [
      "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    const index = Math.round(degrees / 22.5) % 16
    return directions[index]
  }

  return (
    <div className="app">
      <div className="container">
        <div className="header">
          <h1>Predicción de lluvia</h1>
          <button onClick={fetchPrediction} disabled={loading} className="refresh-btn">
            {loading ? (
              <>
                <Loader2 size={16} className="spinner" />
                Cargando...
              </>
            ) : (
              <>
                <RefreshCw size={16} />
                Realizar nueva predicción
              </>
            )}
          </button>
        </div>

        {error && <div className="error-alert">{error}</div>}

        {loading && !prediction && (
          <div className="loading-container">
            <Loader2 size={48} className="spinner" />
            <p>Consiguiendo nueva predicción...</p>
          </div>
        )}

        {prediction && (
          <div className="content">
            <div className="rain-risk-card">
              <div className="icon-container">{getRainIcon(prediction.probabilidad_lluvia)}</div>
              <h2>Riesgo de Lluvia</h2>

              <div className="risk-display">
                <div className={`percentage ${getRiskClass(prediction.probabilidad_lluvia)}`}>
                  {Math.round(prediction.probabilidad_lluvia * 100)}%
                </div>
                <p className="prediction-text">{prediction.prediccion}</p>
              </div>

              <div className="confidence-container">
                <span className="confidence-label">Confidence:</span>
                <span className={`confidence-value ${getConfidenceClass(prediction.confianza)}`}>
                  {prediction.confianza}
                </span>
              </div>
            </div>

            <div className="evidence-card">
              <h3>Basado en:</h3>
              <p className="evidence-description">Condiciones meteorológicas utilizadas en la predicción</p>

              <div className="evidence-grid">
                <div className="evidence-item temperature">
                  <span className="label">Temperatura</span>
                  <span className="value">{prediction.evidencia.temperatura.toFixed(1)}°C</span>
                </div>

                <div className="evidence-item humidity">
                  <span className="label">Humedad</span>
                  <span className="value">{prediction.evidencia.humedad.toFixed(1)}%</span>
                </div>

                <div className="evidence-item wind-speed">
                  <span className="label">Velocidad del viento</span>
                  <span className="value">{prediction.evidencia.viento_vel_m_s.toFixed(2)} m/s</span>
                </div>

                <div className="evidence-item wind-direction">
                  <span className="label">Dirección del viento</span>
                  <span className="value">
                    {formatWindDirection(prediction.evidencia.viento_dir)} ({prediction.evidencia.viento_dir.toFixed(1)}°)
                  </span>
                </div>

                <div className="evidence-item pressure">
                  <span className="label">Presión</span>
                  <span className="value">{prediction.evidencia.presion.toFixed(1)} hPa</span>
                </div>

                <div className="evidence-item cloudiness">
                  <span className="label">Nubosidad</span>
                  <span className="value">{prediction.evidencia.nubosidad.toFixed(1)}%</span>
                </div>

                <div className="evidence-item time-period">
                  <span className="label">Franja horaria</span>
                  <span className="value">{prediction.evidencia.franja_horaria}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App