"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/components/ui/use-toast"
import TrendsSidebar from "@/components/trends-sidebar"
import ResultsDisplay from "@/components/results-display"
import { logger } from "@/lib/logger"

const API_CONFIG = {
  host: process.env.NEXT_PUBLIC_API_HOST,
  port: process.env.NEXT_PUBLIC_API_PORT
}

export default function Home() {
  const [url, setUrl] = useState("")
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState("")
  const [results, setResults] = useState(null)
  const { toast } = useToast()

  const analyzeRecipe = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!url) {
      toast({
        title: "Error",
        description: "Please enter a URL",
        variant: "destructive",
      })
      return
    }

    setLoading(true)
    setProgress(0)
    setResults(null)
    setStatus("Starting analysis...")

    const baseUrl = `http://${API_CONFIG.host}:${API_CONFIG.port}`
    logger.debug('Starting recipe analysis', { url, baseUrl })

    try {
      // Stream Request
      setStatus("Connecting to analysis service...")
      logger.debug('Initiating stream request')
      
      const streamResponse = await fetch(`${baseUrl}/check_recipe_stream?url=${encodeURIComponent(url)}`)
      
      if (!streamResponse.ok) {
        throw new Error(`Stream request failed with status ${streamResponse.status}`)
      }

      logger.debug('Stream connected, reading data...')
      const reader = streamResponse.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          logger.debug('Stream completed')
          break
        }

        const text = decoder.decode(value)
        logger.debug('Received stream chunk', { text })
        
        const lines = text.split('\n')

        for (const line of lines) {
          if (line.trim()) {
            setStatus(line.trim())
            logger.debug('Processing status', { status: line.trim() })
            
            if (line.includes("scraping")) {
              setProgress(25)
              logger.debug('Progress update', { progress: 25 })
            }
            else if (line.includes("Extracting ingredients")) {
              setProgress(50)
              logger.debug('Progress update', { progress: 50 })
            }
            else if (line.includes("Analyzing each ingredient")) {
              setProgress(75)
              logger.debug('Progress update', { progress: 75 })
            }
            else if (line.includes("Finished analysis")) {
              setProgress(100)
              logger.debug('Progress update', { progress: 100 })
            }
          }
        }
      }

      // Get final results
      setStatus("Fetching analysis results...")
      logger.debug('Requesting final results')
      
      const resultResponse = await fetch(`${baseUrl}/check_recipe_result?url=${encodeURIComponent(url)}`)
      
      if (!resultResponse.ok) {
        throw new Error(`Result request failed with status ${resultResponse.status}`)
      }

      const resultData = await resultResponse.json()
      logger.debug('Results received', resultData)
      
      setResults(resultData)
      setStatus("Analysis complete")

      toast({
        title: "Success",
        description: "Recipe analysis complete",
      })
    } catch (error) {
      logger.error('Analysis failed', error as Error, { url, baseUrl })
      setStatus("Analysis failed")
      
      toast({
        title: "Error",
        description: error.message || "Failed to analyze recipe. Please try again.",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
      logger.debug('Analysis process ended')
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-2">
          <Card className="p-6 shadow-lg">
            <form onSubmit={analyzeRecipe} className="space-y-4">
              <div className="space-y-2">
                <h2 className="text-2xl font-bold">Analyze Recipe</h2>
                <p className="text-muted-foreground">
                  Paste a recipe URL to analyze its plant-based compatibility
                </p>
              </div>
              <div className="flex gap-2">
                <Input
                  type="url"
                  placeholder="Enter recipe URL"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  required
                  className="flex-grow"
                  disabled={loading}
                />
                <Button 
                  type="submit" 
                  disabled={loading}
                >
                  {loading ? "Analyzing..." : "Analyze"}
                </Button>
              </div>
            </form>

            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-6 space-y-4"
              >
                <div className="flex justify-between items-center">
                  <p className="text-sm text-muted-foreground">{status}</p>
                  <p className="text-sm text-muted-foreground">{progress}%</p>
                </div>
                <Progress value={progress} />
              </motion.div>
            )}

            {results && <ResultsDisplay results={results} />}
          </Card>
        </div>

        <div className="md:col-span-1">
          <TrendsSidebar />
        </div>
      </div>
    </div>
  )
}