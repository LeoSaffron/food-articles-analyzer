"use client"

import { Card } from "@/components/ui/card"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts"

const mockData = [
  { name: "Plant-based", value: 65 },
  { name: "Low-carb", value: 45 },
  { name: "Gluten-free", value: 35 },
  { name: "Keto", value: 30 },
  { name: "Mediterranean", value: 25 }
]

export default function TrendsSidebar() {
  return (
    <Card className="p-6 shadow-lg">
      <h2 className="text-xl font-bold mb-4">Dietary Trends</h2>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={mockData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" />
            <Tooltip />
            <Bar
              dataKey="value"
              fill="hsl(var(--primary))"
              radius={[0, 4, 4, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-sm text-muted-foreground mt-4">
        Current popularity of dietary preferences based on recent analysis
      </p>
    </Card>
  )
}