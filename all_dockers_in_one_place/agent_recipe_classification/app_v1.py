from fastapi import FastAPI
from pymongo import MongoClient
import requests

# 🔹 Connect to MongoDB
client = MongoClient("mongodb://localhost:27018/")
db = client["foodiesc"]
collection = db["recipes_tasty_co"]

# 🔹 Initialize FastAPI
app = FastAPI()

# 🔹 Query MongoDB for a recipe by URL
def get_recipe_by_url(url):
    return collection.find_one({"url_recipe": url})

# 🔹 Extract a clean list of ingredients
def extract_ingredients(recipe):
    return [ing[1] for ing in recipe.get("ingredients", [])]

# 🔹 Query self-hosted Llama 3 (via Ollama) for ingredient classification
def query_llm(ingredient):
    """Ask Llama3 if an ingredient is plant-based"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": f"Is '{ingredient}' a plant-based ingredient? Respond only with 'Yes' or 'No'.",
        "stream": False
    }
    response = requests.post(url, json=payload)
    reply = response.json()["response"].strip().lower()
    return reply == "yes"  # Convert response to boolean

# 🔹 Check if a recipe is plant-based
def check_recipe(url):
    recipe = get_recipe_by_url(url)
    if not recipe:
        return {"error": "Recipe not found"}

    ingredients = extract_ingredients(recipe)
    plant_based_results = {ing: query_llm(ing) for ing in ingredients}
    all_plant_based = all(plant_based_results.values())

    return {
        "title": recipe["title"],
        "url": recipe["url_recipe"],
        "plant_based": all_plant_based,
        "ingredient_results": plant_based_results
    }

# 🔹 API Endpoint: Check if a recipe is plant-based
@app.get("/check_recipe")
def check_recipe_endpoint(url: str):
    return check_recipe(url)

# 🔹 Run with: uvicorn app:app --reload
