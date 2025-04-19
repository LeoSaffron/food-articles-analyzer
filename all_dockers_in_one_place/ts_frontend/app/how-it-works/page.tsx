import { Card } from "@/components/ui/card"

export default function HowItWorks() {
  return (
    <div className="container mx-auto px-4 py-8">
      <Card className="p-8 shadow-lg">
        <h1 className="text-3xl font-bold mb-6">How Recipe Analyzer Works</h1>
        
        <div className="space-y-6">
          <section>
            <h2 className="text-2xl font-semibold mb-3">The Analysis Process</h2>
            <p className="text-muted-foreground">
              Our recipe analyzer uses advanced natural language processing to break down
              recipes into their component ingredients and analyze them for various
              dietary preferences and restrictions. We help you understand exactly what's
              in your favorite recipes and how they align with different dietary choices.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-3">Steps Involved</h2>
            <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
              <li>Recipe URL submission and validation</li>
              <li>Web scraping to extract recipe content</li>
              <li>Ingredient identification and categorization</li>
              <li>Dietary preference analysis</li>
              <li>Generation of comprehensive results</li>
            </ol>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-3">Usage Guidelines</h2>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>URLs must be less than 150 characters</li>
              <li>Limited to 5 requests per minute</li>
              <li>Only processes publicly accessible recipe pages</li>
              <li>Analysis typically takes 10-15 seconds</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-3">Supported Dietary Preferences</h2>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Vegan - Excludes all animal products</li>
              <li>Plant-based - Focuses on whole plant foods</li>
              <li>Low-carb - Identifies carbohydrate content</li>
              <li>Gluten-free - Flags gluten-containing ingredients</li>
              <li>Mediterranean - Aligns with Mediterranean diet principles</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-3">Privacy & Data</h2>
            <p className="text-muted-foreground">
              We respect your privacy. All recipe analysis is performed in real-time
              and we don't store any personal data or recipe information. Your searches
              and analysis results are not saved or shared with third parties.
            </p>
          </section>
        </div>
      </Card>
    </div>
  )
}