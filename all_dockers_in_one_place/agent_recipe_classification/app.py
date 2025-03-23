from dotenv import load_dotenv
import os
import re
from fastapi import FastAPI
from pymongo import MongoClient
import requests
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request

# 🔹 Load environment variables from .env file
load_dotenv()

# 🔹 Get values from environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27018/")
MONGO_DB = os.getenv("MONGO_DB", "foodiesc")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "recipes_tasty_co")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# 🔹 Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]
misclassified_collection = db["misclassified_ingredients"]

# 🔹 Initialize FastAPI
app = FastAPI()

# Enable CORS (useful for testing from browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Capture all errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(content={"error": str(exc)}, status_code=500)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation error: {exc}")
    return JSONResponse(content={"error": "Invalid input"}, status_code=400)


# 🔹 Query MongoDB for a recipe by URL
def get_recipe_by_url(url):
    return collection.find_one({"url_recipe": url})


# 🔹 Regex-based filtering for symbols, short words, and empty lines
def is_potential_ingredient(ingredient):
    """Basic regex filtering before LLM validation."""

    # Remove empty lines or strings with only special characters
    if not ingredient.strip() or re.match(r"^[^\w]+$", ingredient.strip()):
        return False

    # Remove single characters or symbols
    if len(ingredient.strip()) <= 2:
        return False

    return True


# 🔹 LLM-based validation to check if it's a real ingredient
def validate_ingredient(ingredient):
    """Ask Llama3 if the item is a valid ingredient."""
    # url = "http://localhost:11434/api/generate"LLM_API_URL
    url = LLM_API_URL

    prompt = f"""
    I am extracting ingredients from a recipe. Please determine if the following item is a real food ingredient.

    Examples of valid ingredients:
    - Olive oil
    - Fresh basil
    - Whole wheat flour
    - Almond milk

    Examples of invalid items (not ingredients):
    - "Topping:"
    - "Optional"
    - "For serving"
    - A single punctuation mark like '-'

    **Is the following a valid ingredient?**  
    "{ingredient}"  

    **Respond only with "Yes" or "No". No explanations.**
    """

    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)
    return response.json()["response"].strip().lower() == "yes"

# 🔹 Extract a clean list of valid ingredients
def extract_ingredients(recipe):
    def extract_name_only(item):
        if isinstance(item, dict):
            # Prefer structured 'name' field
            return item.get("name", "").strip()

        elif isinstance(item, list):
            # Heuristic: assume the last element is the name (e.g., ["2 cups", "flour"])
            return str(item[-1]).strip() if item else None

        elif isinstance(item, str):
            return item.strip()

        return None

    raw_items = recipe.get("ingredients", [])
    extracted = [extract_name_only(i) for i in raw_items if extract_name_only(i)]

    # Basic cleanup — skip empty or junk strings
    return [ing for ing in extracted if is_potential_ingredient(ing)]

# 🔹 Extract a clean list of valid ingredients
# def extract_ingredients(recipe):
#     raw_ingredients = [ing[1] for ing in recipe.get("ingredients", [])]
#
#     # Apply regex-based filtering first
#     potential_ingredients = [ing for ing in raw_ingredients if is_potential_ingredient(ing)]
#
#     # Apply LLM-based validation
#     return [ing for ing in potential_ingredients if validate_ingredient(ing)]


# 🔹 Query self-hosted Llama 3 (via Ollama) for ingredient classification
def query_llm(ingredient):
    """Ask Llama3 for plant-based classification with confidence handling and logging."""
    # url = "http://localhost:11434/api/generate"
    url = LLM_API_URL

    prompt = f"""
    I am filtering recipes for vegans. I need to classify ingredients based on how they fit into a plant-based diet.

    There are five possible categories. Choose exactly **one** for the given ingredient:

    1️⃣ **Always Plant-Based** – Ingredients that are inherently vegan, like tofu, soy milk, or explicitly vegan-labeled products (e.g., "vegan sausage").
    2️⃣ **Usually Plant-Based** – Ingredients typically vegan but with some non-vegan variations. Example: Bread (most versions are vegan, but some contain milk or eggs).
    3️⃣ **Check for Plant-Based Version** – Ingredients that exist in both vegan and non-vegan forms, where it’s easy to find a vegan version. Example: Chocolate.
    4️⃣ **Can Be Substituted** – Non-vegan ingredients with clear plant-based alternatives. Example: Milk (can be replaced with soy/oat milk).
    5️⃣ **Not Plant-Based** – Animal-derived ingredients with no easy plant-based alternative. Example: Eggs, Meat, Fish.

    **Classify the following ingredient into exactly one category: "{ingredient}"**  

    **Respond only with one of these labels (no explanations or extra text):**  
    - Always Plant-Based  
    - Usually Plant-Based  
    - Check for Plant-Based Version  
    - Can Be Substituted  
    - Not Plant-Based  
    """

    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)

    llm_response = response.json()["response"].strip()

    # Confidence Handling: If response is unclear, default to "Check for Vegan Version"
    valid_labels = [
        "Always Plant-Based",
        "Usually Plant-Based",
        "Check for Vegan Version",
        "Can Be Substituted",
        "Not Plant-Based"
    ]
    if llm_response not in valid_labels:
        misclassified_collection.insert_one({"ingredient": ingredient, "llm_response": llm_response})
        return "Check for Vegan Version"  # Default fallback

    return llm_response


# 🔹 Check if a recipe is plant-based
def check_recipe(url):
    recipe = get_recipe_by_url(url)
    if not recipe:
        return {"error": "Recipe not found"}

    ingredients = extract_ingredients(recipe)  # Only valid ingredients
    plant_based_results = {ing: query_llm(ing) for ing in ingredients}

    # A recipe is plant-based if all ingredients are either "Always Plant-Based", "Usually Plant-Based", or "Check for Vegan Version"
    all_plant_based = all(result in ["Always Plant-Based", "Usually Plant-Based", "Check for Vegan Version"]
                          for result in plant_based_results.values())

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