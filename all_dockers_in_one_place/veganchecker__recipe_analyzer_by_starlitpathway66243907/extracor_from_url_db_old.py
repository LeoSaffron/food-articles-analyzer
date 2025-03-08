from typing import List, Dict, Any
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import sys
from urllib.parse import quote_plus


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
        """
        Initialize the recipe extractor with MongoDB connection parameters.

        Args:
            host: MongoDB host address
            port: MongoDB port number
            username: MongoDB username (optional)
            password: MongoDB password (optional)
            database: Name of the database
            collection: Name of the collection
        """
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
            print(f"Successfully connected to MongoDB at {self.host}:{self.port}")

        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            sys.exit(1)

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    def extract_ingredients(self, recipe_doc: Dict[Any, Any]) -> List[str]:
        """
        Extract clean ingredient names from a recipe document.

        Args:
            recipe_doc: MongoDB document containing recipe information

        Returns:
            List of clean ingredient names without measurements
        """
        ingredients = recipe_doc.get('ingredients', [])
        clean_ingredients = []

        for ingredient_array in ingredients:
            if len(ingredient_array) < 2:
                continue

            # The ingredient name is typically in the second position
            ingredient_name = ingredient_array[1]

            # Remove HTML spans that contain measurements
            if '<span' in ingredient_name:
                ingredient_name = ingredient_name.split('<span')[0]

            # Remove any trailing commas and whitespace
            ingredient_name = ingredient_name.strip().rstrip(',')

            clean_ingredients.append(ingredient_name)

        return clean_ingredients

    def get_recipe_ingredients(self, url: str) -> List[str]:
        """
        Get ingredients for a specific recipe URL.

        Args:
            url: Recipe URL to search for

        Returns:
            List of ingredients for the recipe
        """
        try:
            recipe = self.collection.find_one({'url_recipe': url})
            if recipe:
                ingredients = self.extract_ingredients(recipe)
                return ingredients
            else:
                print(f"Recipe not found for URL: {url}")
                return []
        except Exception as e:
            print(f"Error retrieving recipe: {e}")
            return []


def main():
    # Example configuration - modify these values as needed
    config = {
        'host': 'localhost',
        'port': 27018,
        'username': None,  # Add username if required
        'password': None,  # Add password if required
        'database': 'foodiesc',
        'collection': 'recipes_tasty_co'
    }

    # Initialize and connect to MongoDB
    extractor = RecipeExtractor(**config)
    extractor.connect()

    # Example URL
    recipe_url = 'https://tasty.co/recipe/homemade-cinnamon-rolls'

    try:
        # Get ingredients
        ingredients = extractor.get_recipe_ingredients(recipe_url)

        # Print results
        if ingredients:
            print(f"\nIngredients for recipe at {recipe_url}:")
            for idx, ingredient in enumerate(ingredients, 1):
                print(f"{idx}. {ingredient}")
        else:
            print("No ingredients found for this recipe")

    except Exception as e:
        print(f"Error processing recipe: {e}")

    finally:
        # Close connection
        extractor.close()


if __name__ == "__main__":
    main()