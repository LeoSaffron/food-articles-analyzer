export default function Footer() {
  return (
    <footer className="border-t">
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="font-semibold mb-2">About</h3>
            <p className="text-sm text-muted-foreground">
              Recipe Analyzer helps you understand the dietary composition of any recipe,
              making it easier to align with your food preferences.
            </p>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Contact</h3>
            <p className="text-sm text-muted-foreground">
              Questions or feedback? Email us at{" "}
              <a
                href="mailto:contact@recipeanalyzer.com"
                className="text-primary hover:underline"
              >
                contact@recipeanalyzer.com
              </a>
            </p>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Legal</h3>
            <p className="text-sm text-muted-foreground">
              © 2024 Recipe Analyzer. All rights reserved.
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}