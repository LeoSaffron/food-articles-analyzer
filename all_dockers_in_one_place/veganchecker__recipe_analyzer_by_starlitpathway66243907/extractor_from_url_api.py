from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pydantic import BaseModel, HttpUrl
import uvicorn
from urllib.parse import quote_plus

app = FastAPI(title="Recipe Ingredients API")


class RecipeExtractor:
    def __init__(
            self,
            host: str = "localhost",
            port: int = 27017,
            username: str = None,
            password: str = None,
            database: str = "recipes",
            collection: str = "recipes",
    ):
        """Initialize the recipe extractor with MongoDB connection parameters."""
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.collection_name = collection
        self.client = None
        self.db = None
        self.collection = None

    def connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            # Create the connection URI
            if self.username and self.password:
                uri = f"mongodb://{quote_plus(self.username)}:{quote_plus(self.password)}@{self.host}:{self.port}"
            else:
                uri = f"mongodb://{self.host}:{self.port}"

            # Connect to MongoDB
            self.client = MongoClient(uri)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]

            # Test connection
            self.client.server_info()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MongoDB connection error: {str(e)}")

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()

    def extract_ingredients(self, recipe_doc: Dict[Any, Any]) -> List[str]:
        """Extract clean ingredient names from a recipe document."""
        ingredients = recipe_doc.get('ingredients', [])
        clean_ingredients = []

        for ingredient_array in ingredients:
            if len(ingredient_array) < 2:
                continue

            ingredient_name = ingredient_array[1]

            if '<span' in ingredient_name:
                ingredient_name = ingredient_name.split('<span')[0]

            ingredient_name = ingredient_name.strip().rstrip(',')
            clean_ingredients.append(ingredient_name)

        return clean_ingredients

    def get_recipe_ingredients(self, url: str) -> List[str]:
        """Get ingredients for a specific recipe URL."""
        try:
            recipe = self.collection.find_one({'url_recipe': url})
            if recipe:
                return self.extract_ingredients(recipe)
            else:
                raise HTTPException(status_code=404, detail="Recipe not found")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Error retrieving recipe: {str(e)}")


# Response model
class IngredientsResponse(BaseModel):
    url: str
    ingredients: List[str]
    total_ingredients: int


@app.get("/ingredients/", response_model=IngredientsResponse)
async def get_ingredients(
        url: HttpUrl = Query(..., description="Recipe URL to fetch ingredients for"),
        db_host: str = Query("localhost", description="MongoDB host"),
        db_port: int = Query(27017, description="MongoDB port"),
        db_name: str = Query("recipes", description="Database name"),
        collection_name: str = Query("recipes", description="Collection name"),
        username: Optional[str] = Query(None, description="MongoDB username (optional)"),
        password: Optional[str] = Query(None, description="MongoDB password (optional)")
):
    """
    Get ingredients for a specific recipe URL with configurable database parameters.

    Returns a list of ingredients and metadata.
    """
    extractor = RecipeExtractor(
        host=db_host,
        port=db_port,
        username=username,
        password=password,
        database=db_name,
        collection=collection_name
    )

    try:
        extractor.connect()
        ingredients = extractor.get_recipe_ingredients(str(url))

        return IngredientsResponse(
            url=str(url),
            ingredients=ingredients,
            total_ingredients=len(ingredients)
        )

    finally:
        extractor.close()


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Recipe Ingredients API"}


def main():
    # Run the server
    uvicorn.run(
        "extractor_from_url_api:app",  # Updated to match the file name
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()