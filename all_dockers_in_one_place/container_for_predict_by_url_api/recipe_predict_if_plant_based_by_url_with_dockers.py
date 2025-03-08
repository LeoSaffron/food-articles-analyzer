import requests
import logging
from flask import Flask, request, jsonify

# -----------------------
# Configuration class
# -----------------------
class Config:
    def __init__(self,
                 # ingredients_service_url="http://localhost:8000/ingredients/",
                 # prediction_service_url="http://localhost:5000/predict",
                 ingredients_service_url="http://host.docker.internal:8000/ingredients/",
                 prediction_service_url="http://host.docker.internal:5000/predict",
                 db_host="host.docker.internal",
                 db_port=27018,
                 db_name="foodiesc",
                 collection_name="recipes_tasty_co",
                 verbosity=logging.DEBUG):
        self.ingredients_service_url = ingredients_service_url
        self.prediction_service_url = prediction_service_url
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.collection_name = collection_name
        self.verbosity = verbosity

        logging.basicConfig(level=verbosity,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("RecipeAnalyzer")

# -----------------------
# Client for Ingredients Service
# -----------------------
class IngredientsServiceClient:
    def __init__(self, config: Config):
        self.config = config

    def get_ingredients(self, recipe_url: str):
        params = {
            "url": recipe_url,
            "db_host": self.config.db_host,
            "db_port": self.config.db_port,
            "db_name": self.config.db_name,
            "collection_name": self.config.collection_name
        }
        try:
            self.config.logger.debug(f"Requesting ingredients from {self.config.ingredients_service_url} with params {params}")
            response = requests.get(self.config.ingredients_service_url, params=params)
            response.raise_for_status()
            data = response.json()
            self.config.logger.debug(f"Ingredients response: {data}")
            return data
        except Exception as e:
            self.config.logger.error(f"Error fetching ingredients: {e}")
            raise

# -----------------------
# Client for Prediction Service
# -----------------------
class PredictionServiceClient:
    def __init__(self, config: Config):
        self.config = config

    def predict(self, ingredient: str):
        payload = {"text": ingredient}
        headers = {"Content-Type": "application/json"}
        try:
            self.config.logger.debug(f"Predicting for '{ingredient}' using {self.config.prediction_service_url}")
            response = requests.post(self.config.prediction_service_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            self.config.logger.debug(f"Prediction response for '{ingredient}': {data}")
            return data
        except Exception as e:
            self.config.logger.error(f"Error predicting for ingredient '{ingredient}': {e}")
            raise

# -----------------------
# Recipe Analyzer that combines both clients
# -----------------------
class RecipeAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.ingredients_client = IngredientsServiceClient(config)
        self.prediction_client = PredictionServiceClient(config)

    def analyze_recipe(self, recipe_url: str):
        result = {"recipe_url": recipe_url, "ingredients": []}
        try:
            ingredients_data = self.ingredients_client.get_ingredients(recipe_url)
            ingredients = ingredients_data.get("ingredients", [])
            if not ingredients:
                self.config.logger.warning("No ingredients found in the response.")
            all_plant_based = True

            for ingredient in ingredients:
                pred_data = self.prediction_client.predict(ingredient)
                # Assuming prediction value 0 means plant based
                prediction_value = pred_data.get("prediction", [None])[0]
                is_plant_based = (prediction_value == 0)
                if not is_plant_based:
                    all_plant_based = False

                result["ingredients"].append({
                    "ingredient": ingredient,
                    "prediction": prediction_value,
                    "is_plant_based": is_plant_based
                })

            result["recipe_is_plant_based"] = all_plant_based
            return result
        except Exception as e:
            self.config.logger.error(f"Error analyzing recipe: {e}")
            return {"error": str(e)}

# -----------------------
# Flask Application Setup
# -----------------------
app = Flask(__name__)

# Create a default configuration instance
default_config = Config()
analyzer = RecipeAnalyzer(default_config)

@app.route('/analyze', methods=['GET'])
def analyze():
    recipe_url = request.args.get('url')
    if not recipe_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    # Optional: override configuration parameters via query string if needed
    ing_url = request.args.get('ingredients_service_url')
    if ing_url:
        default_config.ingredients_service_url = ing_url

    pred_url = request.args.get('prediction_service_url')
    if pred_url:
        default_config.prediction_service_url = pred_url

    db_host = request.args.get('db_host')
    if db_host:
        default_config.db_host = db_host

    db_port = request.args.get('db_port')
    if db_port:
        try:
            default_config.db_port = int(db_port)
        except ValueError:
            default_config.logger.error("Invalid 'db_port' parameter.")
            return jsonify({"error": "Invalid 'db_port' parameter. Must be an integer."}), 400

    db_name = request.args.get('db_name')
    if db_name:
        default_config.db_name = db_name

    collection_name = request.args.get('collection_name')
    if collection_name:
        default_config.collection_name = collection_name

    analysis_result = analyzer.analyze_recipe(recipe_url)
    return jsonify(analysis_result)

# if __name__ == '__main__':
#     # Run on host 0.0.0.0 and port 8086 (or adjust as needed)
#     app.run(debug=True, host="0.0.0.0", port=8086)
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8086)

