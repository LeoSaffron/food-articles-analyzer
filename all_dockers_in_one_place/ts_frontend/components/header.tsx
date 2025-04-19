import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function Header() {
  return (
    <header className="border-b">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="space-y-2">
            <h1 className="text-2xl font-bold">Recipe Analyzer</h1>
            <p className="text-muted-foreground max-w-2xl">
              Discover how recipes align with modern dietary preferences. Whether you're exploring plant-based options, 
              following specific dietary guidelines, or just curious about your favorite recipes' nutritional profile, 
              our analyzer breaks down ingredients and provides detailed insights about their compatibility with various diets.
            </p>
          </div>
          <Button variant="outline" asChild>
            <Link href="/how-it-works">
              How it works
            </Link>
          </Button>
        </div>
      </div>
    </header>
  )
}