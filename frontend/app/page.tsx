"use client"

import { useState } from "react"
import { format } from "date-fns"
import { CalendarIcon } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { WeatherDisplay } from "@/components/weather-display"

export default function WeatherPrediction() {
  const [date, setDate] = useState<Date>(new Date())
  const [timeOfDay, setTimeOfDay] = useState("morning")

  // Mock weather data - would be replaced with API data in a real application
  const weatherData = {
    temperature: 72,
    humidity: 65,
    windVelocity: 8,
    windDirection: "NE",
    pressure: 1012,
    precipitation: 20,
    cloudiness: 30,
  }

  return (
    <div className="container mx-auto py-10">
      <h1 className="text-3xl font-bold mb-8 text-center">Weather Prediction</h1>

      <div className="grid gap-6 md:grid-cols-[300px_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Select Date & Time</CardTitle>
            <CardDescription>Choose when you want to check the weather</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Date</label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant={"outline"}
                    className={cn("w-full justify-start text-left font-normal", !date && "text-muted-foreground")}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {date ? format(date, "PPP") : <span>Pick a date</span>}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    mode="single"
                    selected={date}
                    onSelect={(newDate) => newDate && setDate(newDate)}
                    initialFocus
                  />
                </PopoverContent>
              </Popover>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Time of Day</label>
              <Select value={timeOfDay} onValueChange={setTimeOfDay}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select time of day" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="early-morning">Early Morning</SelectItem>
                  <SelectItem value="morning">Morning</SelectItem>
                  <SelectItem value="noon">Noon</SelectItem>
                  <SelectItem value="evening">Evening</SelectItem>
                  <SelectItem value="night">Night</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button className="w-full mt-4">Update Forecast</Button>
          </CardContent>
        </Card>

        <WeatherDisplay date={date} timeOfDay={timeOfDay} weatherData={weatherData} />
      </div>
    </div>
  )
}
