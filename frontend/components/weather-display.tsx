"use client"

import type React from "react"

import { format } from "date-fns"
import { Cloud, CloudRain, Compass, Droplets, Gauge, Thermometer, Wind } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

interface WeatherData {
  temperature: number
  humidity: number
  windVelocity: number
  windDirection: string
  pressure: number
  precipitation: number
  cloudiness: number
}

interface WeatherDisplayProps {
  date: Date
  timeOfDay: string
  weatherData: WeatherData
}

export function WeatherDisplay({ date, timeOfDay, weatherData }: WeatherDisplayProps) {
  const formattedDate = format(date, "EEEE, MMMM d, yyyy")
  const formattedTimeOfDay = timeOfDay
    .replace("-", " ")
    .replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase())

  // Get background color based on time of day
  const getTimeBackground = () => {
    switch (timeOfDay) {
      case "early-morning":
        return "bg-gradient-to-r from-indigo-900 to-blue-700"
      case "morning":
        return "bg-gradient-to-r from-blue-400 to-cyan-300"
      case "noon":
        return "bg-gradient-to-r from-cyan-400 to-sky-300"
      case "evening":
        return "bg-gradient-to-r from-orange-400 to-pink-500"
      case "night":
        return "bg-gradient-to-r from-slate-900 to-slate-700"
      default:
        return "bg-gradient-to-r from-blue-400 to-cyan-300"
    }
  }

  return (
    <div className="space-y-6">
      <Card className={`${getTimeBackground()} text-white border-none shadow-lg`}>
        <CardHeader>
          <CardTitle className="text-2xl">{formattedDate}</CardTitle>
          <p className="text-lg font-medium">{formattedTimeOfDay}</p>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Thermometer className="h-10 w-10 mr-2" />
              <div>
                <p className="text-4xl font-bold">{weatherData.temperature}°F</p>
              </div>
            </div>
            <div className="text-right">
              <div className="flex items-center justify-end mb-2">
                <CloudRain className="h-6 w-6 mr-1" />
                <span>{weatherData.precipitation}% chance</span>
              </div>
              <div className="flex items-center justify-end">
                <Cloud className="h-6 w-6 mr-1" />
                <span>{weatherData.cloudiness}% cloud cover</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Humidity"
          value={`${weatherData.humidity}%`}
          icon={<Droplets className="h-5 w-5" />}
          progress={weatherData.humidity}
        />

        <MetricCard
          title="Wind"
          value={`${weatherData.windVelocity} mph`}
          icon={<Wind className="h-5 w-5" />}
          subtitle={`Direction: ${weatherData.windDirection}`}
          subtitleIcon={<Compass className="h-4 w-4" />}
        />

        <MetricCard title="Pressure" value={`${weatherData.pressure} hPa`} icon={<Gauge className="h-5 w-5" />} />

        <MetricCard
          title="Precipitation"
          value={`${weatherData.precipitation}%`}
          icon={<CloudRain className="h-5 w-5" />}
          progress={weatherData.precipitation}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Cloudiness</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <Cloud className="h-10 w-10 text-slate-500" />
            <div className="flex-1">
              <Progress value={weatherData.cloudiness} className="h-2" />
            </div>
            <div className="font-medium">{weatherData.cloudiness}%</div>
          </div>
          <div className="mt-4 flex justify-between text-sm text-muted-foreground">
            <span>Clear</span>
            <span>Partly Cloudy</span>
            <span>Overcast</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: string
  icon: React.ReactNode
  subtitle?: string
  subtitleIcon?: React.ReactNode
  progress?: number
}

function MetricCard({ title, value, icon, subtitle, subtitleIcon, progress }: MetricCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          {icon}
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {progress !== undefined && <Progress value={progress} className="h-1 mt-2" />}
        {subtitle && (
          <div className="flex items-center mt-2 text-sm text-muted-foreground">
            {subtitleIcon && <span className="mr-1">{subtitleIcon}</span>}
            {subtitle}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
