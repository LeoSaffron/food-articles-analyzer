import requests
from bs4 import BeautifulSoup
import ollama
import json
import argparse
from unidecode import unidecode
import re
import os

def scrape_webpage(url, verbose=0, cache_file="cached_page.html"):
    """Scrapes a webpage and extracts text content. Uses local cache if available."""
    if os.path.exists(cache_file):
        if verbose >= 1:
            print(f"[INFO] Using cached HTML from: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return None, f"Error: Failed to fetch page (Status Code: {response.status_code})"

        html_content = response.text
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        if verbose >= 1:
            print(f"[INFO] Saved HTML to {cache_file}")

    soup = BeautifulSoup(html_content, "html.parser")
    raw_text = unidecode(soup.get_text(separator="\n", strip=True))

    with open('cached_page2.html', "w", encoding="utf-8") as f:
        f.write(raw_text)

    if verbose >= 1:
        print(f"[INFO] Extracted {len(raw_text)} characters from HTML")

    return raw_text, None

def parse_quantity(quantity_text):
    """Parses ingredient quantity and splits it into quantity and unit."""
    if not quantity_text:
        return None, None

    match = re.match(r"^([\d/.\s]+)\s*([a-zA-Z%]+)?$", quantity_text)
    if not match:
        return None, None

    quantity, unit = match.groups()

    try:
        quantity = eval(quantity) if "/" in quantity or "." in quantity else int(quantity)
    except:
        quantity = None

    return quantity, unit

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

        # Fix common issues like trailing commas
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)

        # Fix missing quotes around string values (e.g., `total: 2.5 hours` → `"total": "2.5 hours"`)
        # json_text = re.sub(r'(:\s*)(\d+(\.\d+)?\s*[a-zA-Z]+)', r'\1"\2"', json_text)
        # Match values after colon that are not quoted and not true/false/null/number → wrap in quotes
        # json_text = re.sub(
        #     r'(:\s*)(?!true|false|null)(?!["\[{])([a-zA-Z][^,\n{}\[\]]+)',
        #     lambda m: f'{m.group(1)}"{m.group(2).strip()}"',
        #     json_text
        # )

        # Fix unquoted string values like 30 minutes, 22g, etc.
        json_text = re.sub(
            r'(:\s*)(?!true|false|null)(?!["\[{])([^\s",\[\]{}][^,\n{}\[\]]*)',
            lambda m: f'{m.group(1)}"{m.group(2).strip()}"'
            if not re.fullmatch(r'[\d.]+', m.group(2).strip()) else m.group(0),
            json_text
        )

        # Fix improperly formatted JSON keys
        json_text = re.sub(r'\\?"([a-zA-Z0-9_-]+)\\?"\s*:', r'"\1":', json_text)

        # 🔧 Fix missing commas between objects or arrays followed by keys
        json_text = re.sub(r'([}\]])\s*\n\s*(")', r'\1,\n\2', json_text)

        # Ensure braces are balanced
        open_braces = json_text.count("{")
        close_braces = json_text.count("}")
        if open_braces > close_braces:
            json_text += "}" * (open_braces - close_braces)
        elif close_braces > open_braces:
            json_text = "{" * (close_braces - open_braces) + json_text

        # **Split JSON into lines for debugging**
        lines = json_text.splitlines()

        # Try parsing and capture exact failing line
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            error_line_number = e.lineno  # Get the line number causing the issue
            error_column = e.colno  # Get the column number
            error_message = str(e)

            print("\n[ERROR] JSON Parsing Failed:")
            print(f"       ↳ {error_message}")

            # **Enumerate and print all lines with line numbers**
            print("\n[DEBUG] Raw JSON Output with Line Numbers:")
            for i, line in enumerate(lines, 1):
                line_prefix = f"  {i:>4} | "  # Right-align line numbers for better readability
                if i == error_line_number:
                    print(f"{line_prefix}{line}")  # Print the problematic line
                    print(f"{' ' * (len(line_prefix) + error_column - 2)}^ ERROR HERE")
                else:
                    print(f"{line_prefix}{line}")

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
        "time": {{
            "prep": "30 minutes",   // always string, even if numeric
            "cook": "string",  // always string, even if numeric
            "total": "1 hour 30 minutes"  // always string, even if numeric
        }},
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

    -----
    Recipe Text:
    {raw_text}
    -----
    """

    if debug or verbose >= 2:
        print("\n[DEBUG] Final prompt sent to LLM:\n")
        print(prompt)

    # response = ollama.chat(
    #     model="llama3:8B",
    #     messages=[{"role": "user", "content": prompt}],
    #     options={"temperature": 0}  # Set model temperature to 0 for consistency
    # )
    response = ollama.chat(
        model="llama3:8B",  # Change to your exact model
        messages=[
            {
                "role": "system",
                "content": """
                You are a structured data extraction assistant. Always return valid JSON.

                **IMPORTANT RULES**:
                - **DO NOT** return `"recipe": {}`. Just return JSON directly.
                - **DO NOT** use `"name"`. Use `"title"` instead.
                - **DO NOT** add extra text like "Here is the extracted JSON".
                - **Ensure `"quantity"` is separated from `"unit"`** (e.g., `"4 oz"` → `"quantity": 4, "unit": "oz"`).
                - **break down amounts when both numerical values and units of measurement are present** (e.g., `"prep_time" : "15 minutes"` → `"prep_time": {"value" : 15, "unit": "minutes"}`).
                - **Put every numerical value that contains text in "" (e.g., instead of `15 hours`, return `"15 hours"`).**
                - **If an ingredient has no `"quantity"` or `"unit"`, remove those keys.**
                - **DO NOT** include Markdown formatting (` ```json ` or backticks).
                - **If the title is missing, return `"title": "Unknown Recipe"` instead of an empty string.**
                - **Return only valid JSON. No extra text, no code blocks, no explanations.**
                """
            },
            {"role": "user", "content": prompt}  # Your actual query
        ],
        options={"temperature": 0}  # Keep deterministic
    )

    raw_output = response["message"]["content"].strip()

    if debug or verbose >= 2:
        print("\n[DEBUG] Raw LLM Output:\n", raw_output)

    structured_data = extract_json_from_text(raw_output, verbose=2)

    if "title" not in structured_data or not structured_data["title"]:
        structured_data["title"] = "Unknown Recipe"  # Ensure title is always present

    # if "ingredients" in structured_data:
    #     for ingredient in structured_data["ingredients"]:
    #         if "quantity" in ingredient:
    #             quantity, unit = parse_quantity(ingredient["quantity"])
    #             if quantity is not None:
    #                 ingredient["quantity"] = quantity
    #             if unit:
    #                 ingredient["unit"] = unit
    #             del ingredient["quantity"]
    #
    #         if "category" in ingredient and not ingredient["category"]:
    #             del ingredient["category"]

    if "ingredients" in structured_data:
        for ingredient in structured_data["ingredients"]:
            # Only parse if quantity is a standalone number + unit
            q = ingredient.get("quantity")
            if q and isinstance(q, str) and re.match(r"^[\d\s/\.]+[a-zA-Z%]*$", q.strip()):
                quantity, unit = parse_quantity(q.strip())
                if quantity is not None:
                    ingredient["quantity"] = quantity
                if unit:
                    ingredient["unit"] = unit
                # Only delete original quantity if it was converted
                if quantity is not None or unit:
                    del ingredient["quantity"]

            if "category" in ingredient and not ingredient["category"]:
                del ingredient["category"]

    if verbose >= 1:
        print("[INFO] Successfully parsed recipe data.")

    return structured_data

def get_recipe(url, debug=False, verbose=0):
    """Scrapes a recipe webpage and extracts structured data using LLM."""
    raw_text, error = scrape_webpage(url, verbose)
    if error:
        return {"error": error}

    structured_recipe = parse_recipe_with_llm(raw_text, debug, verbose)
    # structured_recipe["source"] = url  # Add source URL
    # Move 'url' to be the first key in the resulting dict
    structured_recipe = {
        "url": url,
        **structured_recipe
    }

    return structured_recipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape and parse recipes using LLM.")
    parser.add_argument("url", type=str, help="Recipe URL to scrape")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (shows raw LLM output)")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0, help="Verbosity level (0: default, 1: info, 2: detailed)")

    args = parser.parse_args()
    recipe_data = get_recipe(args.url, debug=args.debug, verbose=args.verbose)

    print(json.dumps(recipe_data, indent=4, ensure_ascii=False))