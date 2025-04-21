import { Card } from "@/components/ui/card"

const mockData = [
  { name: "Plant-based", value: 65 },
  { name: "Low-carb", value: 45 },
  { name: "Gluten-free", value: 35 },
]

export default function TrendsSidebarFallback() {
  return (
    <Card className="p-6 shadow-lg">
      <h2 className="text-xl font-bold mb-4">Dietary Trends</h2>
      <div className="space-y-4">
        {mockData.map((item, index) => (
          <div key={index} className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>{item.name}</span>
              <span>{item.value}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
              <div
                className="bg-primary h-2.5 rounded-full"
                style={{ width: `${item.value}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
      <p className="text-sm text-muted-foreground mt-4">
        Current popularity of dietary preferences
      </p>
    </Card>
  )
}