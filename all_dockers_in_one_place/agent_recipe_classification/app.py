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
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.responses import StreamingResponse
from time import sleep
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from collections import defaultdict
from recipe_scrapers import scrape_me



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

# Store last result by URL (you can improve this later)
last_result_by_url = {}

class StreamToClientHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        msg = self.format(record)
        self.messages.append(msg)

    def get_stream(self):
        for msg in self.messages:
            yield f"data: {msg}\n\n"  # Server-Sent Events format

stream_handler = StreamToClientHandler()
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)

stream_logger = logging.getLogger("stream_logger")
stream_logger.setLevel(logging.INFO)
stream_logger.addHandler(stream_handler)


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
def query_llm(ingredient, debug=False, verbose=0):
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

    if debug or verbose >= 2:
        print(f"\n[DEBUG] Ingredient classification: {ingredient}. Final prompt sent to LLM:\n")
        print(prompt)

    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)

    llm_response = response.json()["response"].strip()

    if debug or verbose >= 2:
        print("\n[DEBUG] Ingredient classification: {ingredient}. Output LLM:\n")
        print(llm_response)

    # Confidence Handling: If response is unclear, default to "Check for Vegan Version"
    valid_labels = [
        "Always Plant-Based",
        "Usually Plant-Based",
        "Check for Plant-Based Version",
        "Can Be Substituted",
        "Not Plant-Based"
    ]
    if llm_response not in valid_labels:
        misclassified_collection.insert_one({"ingredient": ingredient, "llm_response": llm_response})
        return "Check for Plant-Based Version"  # Default fallback

    return llm_response


# 🔹 Check if a recipe is plant-based
def check_recipe(url, debug=True, log_callback=None):

    def log(msg):
        if log_callback:
            log_callback(msg + '\n')
        logging.info(msg)

    recipe = get_recipe_by_url(url)

    if not recipe:
        logging.info(f"[INFO] Recipe not found in database: {url}")
        print(f"[INFO] Recipe not found in database: {url}")
        yield f"[INFO] Recipe not found in database: {url}"
        log(f"[INFO] Recipe not found in database: {url}")

        logging.info(f"[INFO] Scraping recipe from URL: {url}")
        print(f"[INFO] Scraping recipe from URL: {url}")
        yield (f"[INFO] Scraping recipe from URL: {url}")

        # scraper = RecipeScraper(url)
        # scraped_recipe = scraper.get_recipe(url)
        scraper = RecipeScraper(url, mongo_uri=MONGO_URI, debug=debug, verbose=2 if debug else 0)
        try:
            logging.info("[INFO] Proceeding to Scrape recipe via recipe_scrapers")
            yield "[INFO] Proceeding to Scrape recipe via recipe_scrapers"
            rs = scrape_me(url)
            scraped_recipe = {
                "title": rs.title(),
                "ingredients": rs.ingredients(),
                "instructions": [rs.instructions()],
                "url_recipe": url
            }
            logging.info("[INFO] Scraped recipe via recipe_scrapers")
            yield "[INFO] Scraped recipe via recipe_scrapers\n"
                # yield f"{scraped_recipe}\n"
                #
                # yield "\n"
                # yield f'is_valid_recipe(scraped_recipe)         {is_valid_recipe(scraped_recipe)}       '
                # yield f'isinstance(recipe_data, dict)             {isinstance(scraped_recipe, dict)}               '
                # # yield f'"error" not in recipe_data{"error" not in scraped_recipe}'
                # yield f'"title" in recipe_data and recipe_data["title"].strip()          {"title" in scraped_recipe and bool(scraped_recipe["title"].strip())}            '
                # yield f'"ingredients" in recipe_data and isinstance(recipe_data["ingredients"], list) and recipe_data[ \
                #     "ingredients"]            {"ingredients" in scraped_recipe}            '
                # yield f'"instructions" in scraped_recipe and isinstance(scraped_recipe["instructions"], list) and scraped_recipe["instructions"]                   {"instructions" in scraped_recipe and isinstance([scraped_recipe["instructions"]], list) and bool(scraped_recipe["instructions"])}               '
                # yield '                                       '
                # yield f'{"ingredients" in scraped_recipe and isinstance(scraped_recipe["ingredients"], list) and bool(scraped_recipe["ingredients"])}'
                # yield f'{"instructions" in scraped_recipe and isinstance(scraped_recipe["instructions"], list) and bool(scraped_recipe["instructions"])}'
                # yield f'{isinstance(scraped_recipe["instructions"], list)}'

            if is_valid_recipe(scraped_recipe):
                collection.insert_one(scraped_recipe)
                recipe = scraped_recipe
            else:
                raise ValueError("Invalid schema from recipe_scrapers")
        except Exception as e:
            yield "Could not scrape with dedicated library, proceeding with the agent"
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
        yield f"[INFO] Recipe found in database for URL: {url}"
        print(f"[INFO] Recipe found in database for URL: {url}")
        log(f"[INFO] Recipe found in database for URL: {url}")

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

# def check_recipe_streaming(url: str):
#     yield f"[INFO] Checking recipe URL: {url}"
#     yield "[INFO] Looking up in database..."
#     yield "[INFO] Not found. Scraping..."
#     yield "[INFO] Running analysis..."
#     yield "[INFO] Finished!"

# Global in-memory result cache
result_cache = {}

# def check_recipe_streaming(url, debug=True):
#     def log(msg):
#         logging.info(msg)
#         yield msg + '\n'
#
#     # Check DB first
#     recipe = get_recipe_by_url(url)
#     if not recipe:
#         yield from log(f"[INFO] Recipe not found in database: {url}")
#         yield from log(f"[INFO] Scraping recipe from URL: {url}")
#
#         scraper = RecipeScraper(url, mongo_uri=MONGO_URI, debug=debug, verbose=2 if debug else 0)
#         scraped_recipe = scraper.get_recipe(url, debug=debug, verbose=2 if debug else 0)
#
#         if is_valid_recipe(scraped_recipe):
#             yield from log("[INFO] Scraped recipe is valid, saving to MongoDB")
#             scraper.save_to_mongodb(scraped_recipe)
#             recipe = scraped_recipe
#         else:
#             yield from log("[WARN] Scraped recipe is invalid.")
#             yield from log(str(scraped_recipe))
#             result_cache[url] = {"error": "Recipe not found and could not be scraped."}
#             return
#
#     else:
#         yield from log(f"[INFO] Recipe found in database for URL: {url}")
#
#     yield from log("[INFO] Extracting ingredients and running analysis...")
#
#     ingredients = extract_ingredients(recipe)
#     plant_based_results = {ing: query_llm(ing) for ing in ingredients}
#
#     all_plant_based = all(result in [
#         "Always Plant-Based",
#         "Usually Plant-Based",
#         "Check for Vegan Version"
#     ] for result in plant_based_results.values())
#
#     final_result = {
#         "title": recipe.get("title", "Unknown Recipe"),
#         "url": recipe.get("url_recipe", url),
#         "plant_based": all_plant_based,
#         "ingredient_results": plant_based_results
#     }
#
#     # ✅ Save to cache
#     result_cache[url] = final_result
#     yield from log("[INFO] Finished analysis.")
#
# @app.get("/check_recipe_stream")
# async def stream_logs(url: str):
#     def log_generator():
#         yield "Starting analysis for URL: " + url + "\n"
#         result = None
#
#         # Replace check_recipe with a generator version or yield steps manually
#         for line in check_recipe_streaming(url):  # <-- You need to define this generator
#             yield line + "\n"
#
#         yield "[Log stream ended]\n"
#
#     return StreamingResponse(log_generator(), media_type="text/plain")

from fastapi.responses import StreamingResponse

@app.get("/check_recipe_stream")
def check_recipe_stream(url: str):
    def event_stream():
        yield_msgs = []
        def log_callback(msg):
            yield_msgs.append(msg)
            yield msg  # Optional if you want immediate flush too
        yield f"[INFO] Checking recipe URL: {url}\n"
        recipe = get_recipe_by_url(url)

        if not recipe:
            yield "[INFO] Recipe not found in DB, scraping...\n"
            # scraper = RecipeScraper(url, mongo_uri=MONGO_URI, debug=True, verbose=2)
            # for msg in scraper.get_recipe(url, debug=True, verbose=2, log_callback=log_callback):
            #     yield msg + "\n"
            # # recipe = scraper.get_recipe(url)
            #
            # log_buffer = []
            #
            # def log_cb(msg):
            #     log_buffer.append(msg)
            scraper = RecipeScraper(url, mongo_uri=MONGO_URI, debug=True, verbose=2)
            try:
                logging.info("[INFO] Proceeding to Scrape recipe via recipe_scrapers")
                rs = scrape_me(url)
                recipe_data = {
                    "title": rs.title(),
                    "ingredients": rs.ingredients(),
                    "instructions": rs.instructions(),
                    "url_recipe": url
                }
                logging.info("[INFO] Scraped recipe via recipe_scrapers")
                if is_valid_recipe(recipe_data):
                    collection.insert_one(recipe_data)
                    recipe = recipe_data
                else:
                    raise ValueError("Invalid schema from recipe_scrapers")
            except Exception as e:
                yield from scraper.get_recipe(url, debug=True, verbose=2, log_callback=log_callback)
                recipe = scraper.result_get_recipe

            # for msg in log_buffer:
            #     yield f"{msg}\n"




            if not is_valid_recipe(recipe):
                yield "[ERROR] Invalid recipe, aborting.\n"
                return
            scraper.save_to_mongodb(recipe)
        else:
            yield "[INFO] Recipe found in database.\n"

        yield "[INFO] Extracting ingredients...\n"
        ingredients = extract_ingredients(recipe)
        yield "[INFO] Successfully Extracted ingredient list. Analyzing each ingredient...\n"
        result_map = {ing: query_llm(ing) for ing in ingredients}

        yield "[INFO] Finished analysis.\n"

        last_result_by_url[url] = {
            "title": recipe.get("title", "Unknown Recipe"),
            "url": recipe.get("url_recipe", url),
            "plant_based": all(r in [
                "Always Plant-Based", "Usually Plant-Based", "Check for Vegan Version"
            ] for r in result_map.values()),
            "ingredient_results": result_map
        }
        # sleep(2)
        # yield "test 1\n"
        # sleep(2)
        # yield "test 2\n"
        # sleep(2)
        # yield "test 3\n"
        # sleep(2)
        # yield "test 4\n"
    return StreamingResponse(event_stream(), media_type="text/plain")






# def stream_logs_endpoint(url: str):
#     def event_stream():
#         yield "Starting log stream...\n"
#
#         def send(line):
#             # This gets called with each log line
#             yield_line = line if line.endswith("\n") else line + "\n"
#             yield yield_line
#
#         buffer = []
#
#         def log_to_buffer(message):
#             buffer.append(message + "\n")
#
#         result = check_recipe(url, debug=True, log_callback=log_to_buffer)
#
#         for line in buffer:
#             yield line
#             sleep(0.1)  # slight delay for smoother UI updates
#
#         yield "[Log stream ended]\n"
#
#     return StreamingResponse(event_stream(), media_type="text/plain")

# 🔹 API Endpoint: Check if a recipe is plant-based
@app.get("/check_recipe")
def check_recipe_endpoint(url: str):
    return check_recipe(url)

# @app.get("/check_recipe_result")
# def check_recipe_result(url: str):
#     result = check_recipe(url, debug=False, log_callback=None)
#     return result

@app.get("/check_recipe_result")
def get_recipe_result(url: str):
    result = last_result_by_url.get(url)
    if not result:
        return {"error": "No result available for this URL. Please run analysis first."}
    return result

# @app.get("/stream_logs")
# async def stream_logs(url: str):
#     async def log_generator():
#         from io import StringIO
#         import sys
#
#         buffer = StringIO()
#         handler = logging.StreamHandler(buffer)
#         handler.setLevel(logging.INFO)
#
#         logger = logging.getLogger("uvicorn.error")
#         logger.addHandler(handler)
#
#         # Run recipe check in background
#         loop = asyncio.get_event_loop()
#         result_future = loop.run_in_executor(None, check_recipe, url)
#
#         last_pos = 0
#         while not result_future.done():
#             await asyncio.sleep(0.5)
#             buffer.seek(last_pos)
#             chunk = buffer.read()
#             last_pos = buffer.tell()
#             if chunk:
#                 yield f"data: {chunk.strip()}\n\n"
#
#         # Final logs
#         buffer.seek(last_pos)
#         chunk = buffer.read()
#         if chunk:
#             yield f"data: {chunk.strip()}\n\n"
#
#         logger.removeHandler(handler)
#
#     return StreamingResponse(log_generator(), media_type="text/event-stream")

async def stream_logs(url: str):
    def log_generator():
        yield f"Starting analysis for URL: {url}\n"
        for line in check_recipe_streaming(url):
            yield line + "\n"
        yield "[Log stream ended]\n"

    return StreamingResponse(log_generator(), media_type="text/plain")

# 🔹 Run with: uvicorn app:app --reload