"use client"

import { motion } from "framer-motion"
import { CheckCircle, XCircle, AlertCircle } from "lucide-react"

interface IngredientResults {
  [key: string]: string
}

interface Result {
  title: string
  url: string
  plant_based: boolean
  ingredient_results: IngredientResults
}

interface ResultsDisplayProps {
  results: Result
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case "Always Plant-Based":
      return <CheckCircle className="text-green-500" size={20} />
    case "Usually Plant-Based":
      return <CheckCircle className="text-green-400" size={20} />
    case "Check for Plant-Based Version":
      return <AlertCircle className="text-yellow-500" size={20} />
    case "Not Plant-Based":
      return <XCircle className="text-red-500" size={20} />
    default:
      return <AlertCircle className="text-gray-500" size={20} />
  }
}

const getStatusColor = (status: string) => {
  switch (status) {
    case "Always Plant-Based":
      return "bg-green-100 dark:bg-green-900/20"
    case "Usually Plant-Based":
      return "bg-green-50 dark:bg-green-900/10"
    case "Check for Plant-Based Version":
      return "bg-yellow-50 dark:bg-yellow-900/10"
    case "Not Plant-Based":
      return "bg-red-50 dark:bg-red-900/10"
    default:
      return "bg-gray-50 dark:bg-gray-900/10"
  }
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-8 space-y-6"
    >
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">{results.title}</h2>
        <div className={`p-4 rounded-lg ${results.plant_based ? 'bg-green-100 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/10'}`}>
          <div className="flex items-center gap-2">
            {results.plant_based ? (
              <CheckCircle className="text-green-500" />
            ) : (
              <XCircle className="text-red-500" />
            )}
            <span className="font-medium">
              {results.plant_based ? "Recipe is Plant-Based" : "Recipe is Not Plant-Based"}
            </span>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Ingredients Analysis</h3>
        <div className="space-y-2">
          {Object.entries(results.ingredient_results).map(([ingredient, status], index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-3 rounded-lg flex items-center justify-between ${getStatusColor(status)}`}
            >
              <div>
                <p className="font-medium capitalize">{ingredient}</p>
                <p className="text-sm text-muted-foreground">
                  {status}
                </p>
              </div>
              {getStatusIcon(status)}
            </motion.div>
          ))}
        </div>
      </div>

      <div className="text-sm text-muted-foreground">
        <p>Original recipe: <a href={results.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">{results.url}</a></p>
      </div>
    </motion.div>
  )
}