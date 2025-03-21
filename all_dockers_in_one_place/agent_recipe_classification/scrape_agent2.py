import requests
from bs4 import BeautifulSoup
import ollama
import json
import argparse
from unidecode import unidecode
import os
import hashlib
from requests_html import HTMLSession
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

from bs4 import BeautifulSoup

def clean_html(raw_html, keep_tags=None):
    """Strip style/script/meta and return readable text from rendered HTML."""
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "noscript", "meta", "header", "footer", "nav"]):
        tag.decompose()

    # Optionally remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove attributes like class/id/etc if you want to simplify
    for tag in soup.find_all():
        tag.attrs = {}

    # Optionally: Extract only inside <main> or a known content div
    main = soup.find("main") or soup.body
    text = main.get_text(separator="\n", strip=True)

    return text


import re

def scrape_webpage(url, verbose=0, force_refresh=False):
    """Scrapes a webpage using Playwright and returns clean visible text."""
    os.makedirs(".cache", exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = Path(f".cache/{url_hash}.txt")

    if cache_file.exists() and not force_refresh:
        if verbose >= 1:
            print(f"[INFO] Using cached text from {cache_file}")
        return cache_file.read_text(encoding="utf-8"), None

    try:
        if verbose >= 1:
            print(f"[INFO] Launching headless browser for {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            page.wait_for_selector("h1", timeout=10000)
            html = page.content()
            browser.close()

        clean_text = clean_html(html)
        cache_file.write_text(clean_text, encoding="utf-8")

        if verbose >= 1:
            print(f"[INFO] Page scraped and cleaned ({len(clean_text)} chars)")

        return clean_text, None

    except Exception as e:
        return None, f"Error during fetch/render: {e}"

import re
import json

import re
import json

import re
import json

import re
import json

import re
import json

import re
import json

import re
import json

import re
import json

import re
import json

def extract_json_from_text(text, verbose=0):
    """Extracts valid JSON from LLM output while ensuring correctness."""
    try:
        # Locate first valid JSON object
        first_brace = text.find("{")
        last_brace = text.rfind("}")  # Last valid closing brace

        if first_brace == -1 or last_brace == -1:
            raise ValueError("No valid JSON found in LLM output")

        json_text = text[first_brace:last_brace+1]

        # 🧼 Clean up trailing garbage or second JSON blocks
        json_text = re.sub(r"}\s*{.*", "}", json_text, flags=re.DOTALL)

        # Fix common issues like trailing commas
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)

        # Fix missing quotes around string values (e.g., total: 2.5 hours → "total": "2.5 hours")
        json_text = re.sub(r'(:\s*)(\d+(\.\d+)?\s*[a-zA-Z]+)', r'\1"\2"', json_text)

        # Fix improperly formatted JSON keys
        json_text = re.sub(r'\\?"([a-zA-Z0-9_-]+)\\?"\s*:', r'"\1":', json_text)

        # Split JSON into lines for better debugging
        lines = json_text.splitlines()

        if verbose >= 1:
            print("\n[DEBUG] Full JSON Output (with Line Numbers):")
            for i, line in enumerate(lines, 1):
                print(f"  {i:>4} | {line}")

        # Try parsing the cleaned JSON
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            error_line_number = e.lineno
            error_column = e.colno
            error_message = str(e)

            print("\n[ERROR] JSON Parsing Failed:")
            print(f"       ↳ {error_message}")

            # Highlight the exact error in the debug output
            print("\n[DEBUG] JSON Output with Error Highlighted:")
            for i, line in enumerate(lines, 1):
                prefix = f"  {i:>4} | "
                if i == error_line_number:
                    print(f"{prefix}{line}")
                    print(" " * (len(prefix) + error_column - 1) + "^ ERROR HERE")
                else:
                    print(f"{prefix}{line}")

            return {"error": "Invalid JSON output from LLM", "details": error_message}

    except ValueError as e:
        print("\n[ERROR] JSON Extraction Failed:")
        print(f"       ↳ {e}")
        return {"error": "Invalid JSON output from LLM", "details": str(e)}


def parse_recipe_with_llm(raw_text, debug=False, verbose=0):
    """Sends raw recipe text to LLM and ensures a valid JSON response."""
    prompt = f"""
    Extract structured data from the following recipe text.

    Return JSON formatted like this, with NO extra text before or after:
    {{
        "title": "Recipe Title",  # Use "title" instead of "name". Never return "recipe": {{}}
        "description": "Short description",
        "prep time" : "9 hours"
        "cookTime": "9 hours",
        "ingredients": [
            {{
                "name": "Ingredient",
                "quantity": Number,
                "unit": "Unit (cup, teaspoon, gram)",
                "category": "Filling, Frosting, etc."  # Include only if applicable
            }}
        ],
        "instructions": ["Step 1", "Step 2"]
    }}

    **IMPORTANT RULES**:
    - **DO NOT** return `"recipe": {{}}`. Just return JSON directly.
    - **DO NOT** use `"name"`. Use `"title"` instead.
    - **Put every numerical values that contains text in "" (e.g. instead of 15 hours it should be "15 hours")**
    - **DO NOT** add extra text like "Here is the extracted JSON".
    - **Ensure `"quantity"` is separated from `"unit"`** (e.g., `"4 oz"` → `"quantity": 4, "unit": "oz"`).
    - **If an ingredient has no `"quantity"` or `"unit"`, remove those keys.**
    - **DO NOT** include Markdown formatting (` ```json ` or backticks).
    - **If the title is missing, return `"title": "Unknown Recipe"` instead of an empty string.**
    - **Return only valid JSON. No extra text, no code blocks, no explanations.**

    -----
    Recipe Text:
    {raw_text}
    -----
    """

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}  # Set model temperature to 0 for consistency
    )

    raw_output = response["message"]["content"].strip()

    if debug or verbose >= 2:
        print("\n[DEBUG] Raw LLM Output:\n", raw_output)

    structured_data = extract_json_from_text(raw_output, verbose=2)

    if "title" not in structured_data or not structured_data["title"]:
        structured_data["title"] = "Unknown Recipe"  # Ensure title is always present

    if "ingredients" in structured_data:
        for ingredient in structured_data["ingredients"]:
            if "quantity" in ingredient:
                quantity, unit = parse_quantity(ingredient["quantity"])
                if quantity is not None:
                    ingredient["quantity"] = quantity
                if unit:
                    ingredient["unit"] = unit
                del ingredient["quantity"]

            if "category" in ingredient and not ingredient["category"]:
                del ingredient["category"]

    if verbose >= 1:
        print("[INFO] Successfully parsed recipe data.")

    return structured_data

def get_recipe(url, debug=False, verbose=0):
    """Scrapes a recipe webpage and extracts structured data using LLM."""
    # raw_text, error = scrape_webpage(url, verbose)
    raw_text, error = scrape_webpage(args.url, verbose=args.verbose, force_refresh=args.refresh)

    if error:
        return {"error": error}

    structured_recipe = parse_recipe_with_llm(raw_text, debug, verbose)
    structured_recipe["source"] = url  # Add source URL

    return structured_recipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape and parse recipes using LLM.")
    parser.add_argument("url", type=str, help="Recipe URL to scrape")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (shows raw LLM output)")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0, help="Verbosity level (0: default, 1: info, 2: detailed)")
    parser.add_argument("--refresh", action="store_true", help="Force re-download from website")


    args = parser.parse_args()
    recipe_data = get_recipe(args.url, debug=args.debug, verbose=args.verbose)

    print(json.dumps(recipe_data, indent=4, ensure_ascii=False))