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
from html import unescape

from scrape_agent import RecipeScraper, is_valid_recipe

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

# 🔹 Enable logging to console
logging.basicConfig(level=logging.INFO)

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
# def extract_ingredients(recipe):
#     def extract_name_only(item):
#         if isinstance(item, dict):
#             # Prefer structured 'name' field
#             return item.get("name", "").strip()
#
#         elif isinstance(item, list):
#             # Heuristic: assume the last element is the name (e.g., ["2 cups", "flour"])
#             return str(item[-1]).strip() if item else None
#
#         elif isinstance(item, str):
#             return item.strip()
#
#         return None
#
#     raw_items = recipe.get("ingredients", [])
#     extracted = [extract_name_only(i) for i in raw_items if extract_name_only(i)]
#
#     # Basic cleanup — skip empty or junk strings
#     return [ing for ing in extracted if is_potential_ingredient(ing)]

def extract_ingredients(recipe):
    def flatten(items):
        for i in items:
            if isinstance(i, list):
                yield from flatten(i)
            else:
                yield i

    def clean_html(text):
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        return unescape(text.strip())  # Decode HTML entities

    def extract_from_string(line):
        if not isinstance(line, str):
            return None
        line = clean_html(line)
        if not is_potential_ingredient(line):
            return None
        # Remove units/quantities and notes
        line = re.sub(r"^[\d¼½¾¾\s/.,ozcupsingteabltablespoons]+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\([^)]*\)", "", line)  # Remove anything in brackets
        line = re.sub(r"[,–-].*$", "", line)  # Remove descriptors after comma/dash
        return line.strip()

    def extract_from_list(lst):
        if all(isinstance(i, str) for i in lst):
            # Join list into a sentence, strip html, extract
            combined = " ".join(lst)
            return [extract_from_string(combined)]
        else:
            # Nested structure like sub-ingredients
            return [process_ingredient_item(item) for item in lst]

    def extract_name(item):
        if not isinstance(item, dict):
            return None
        for key in ["ingredient_name", "name"]:
            val = item.get(key)
            if val and is_potential_ingredient(val):
                return clean_html(val)
        return None

    def process_ingredient_item(item):
        if isinstance(item, str):
            return [extract_from_string(item)]
        elif isinstance(item, list):
            return extract_from_list(item)
        elif isinstance(item, dict):
            if "sub-ingredients" in item:
                return [process_ingredient_item(sub) for sub in item["sub-ingredients"]]
            name = extract_name(item)
            return [name] if name else []
        return []

    # Source priority
    raw_items = (
        recipe.get("ingredients") or
        recipe.get("recipeIngredient") or
        []
    )

    nested = [process_ingredient_item(item) for item in raw_items]
    flat = list(flatten(nested))
    cleaned = [i for i in flat if i and is_potential_ingredient(i)]
    return list(dict.fromkeys(cleaned))  # remove duplicates while preserving order

# Simple filter for valid ingredient names
def is_potential_ingredient(text):
    if not text:
        return False
    text = text.strip().lower()
    banned = {
        "toppings", "frosting", "stuffings",
        "dough base", "batter", "glaze", "optional"
    }
    return not any(text == b for b in banned) and len(text) > 2


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
def check_recipe(url, debug=True):
    recipe = get_recipe_by_url(url)

    if not recipe:
        logging.info(f"[INFO] Recipe not found in database: {url}")
        print(f"[INFO] Recipe not found in database: {url}")

        logging.info(f"[INFO] Scraping recipe from URL: {url}")
        print(f"[INFO] Scraping recipe from URL: {url}")

        # scraper = RecipeScraper(url)
        # scraped_recipe = scraper.get_recipe(url)
        scraper = RecipeScraper(url, mongo_uri=MONGO_URI, debug=debug, verbose=2 if debug else 0)
        scraped_recipe = scraper.get_recipe(url, debug=debug, verbose=2 if debug else 0)

        if is_valid_recipe(scraped_recipe):
            logging.info("[INFO] Scraped recipe is valid, saving to MongoDB")
            scraper.save_to_mongodb(scraped_recipe)
            recipe = scraped_recipe
        else:
            logging.warning("[WARN] Scraped recipe is invalid:")
            logging.warning(scraped_recipe)  # ← This will print the failed payload
            return {"error": "Recipe not found and could not be scraped."}

    else:
        logging.info(f"[INFO] Recipe found in database for URL: {url}")
        print(f"[INFO] Recipe found in database for URL: {url}")

    ingredients = extract_ingredients(recipe)
    plant_based_results = {ing: query_llm(ing) for ing in ingredients}

    all_plant_based = all(result in [
        "Always Plant-Based",
        "Usually Plant-Based",
        "Check for Vegan Version"
    ] for result in plant_based_results.values())

    return {
        "title": recipe.get("title", "Unknown Recipe"),
        "url": recipe.get("url_recipe", url),
        "plant_based": all_plant_based,
        "ingredient_results": plant_based_results
    }

# 🔹 API Endpoint: Check if a recipe is plant-based
@app.get("/check_recipe")
def check_recipe_endpoint(url: str):
    return check_recipe(url)

# 🔹 Run with: uvicorn app:app --reload