# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:09:29 2024

@author: hor
"""

# import warnings

# # filter out warnings
# warnings.filterwarnings('ignore')

# # dataframes and vectors
# import pandas as pd
# import numpy as np

# # vectorizing sentences
# from sentence_transformers import SentenceTransformer

# from sklearn.svm import SVC
# import joblib
# from joblib import load

# model_vectorize = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# model_svc = load("trained_models/SVM.pkl")

# print(model_svc.predict([model_vectorize.encode('fried soyballs')]))

##-----------------------------------------------------------------------------


# from flask import Flask, request, jsonify
# import pickle

# class Embedder:
#     def __init__(self):
#         self.model = None

#     def encode(self, sentence):
#         return [sentence]  # Simplified encoding for demonstration purposes only

# class Predictor:
#     def __init__(self):
#         self.model = pickle.load(open("trained_models/LinearSVC.pkl", "rb"))

#     def predict(self, vectorized_sentence):
#         predicted_label = self.model.predict([vectorized_sentence])[0]
#         return {'prediction': int(predicted_label)}

# class TextClassifier:
#     def __init__(self):
#         self.embedder = Embedder()
#         self.predictor = Predictor()

#     @Flask.route('/predict', methods=['POST'])
#     def predict_endpoint(self):
#         sentence = request.get_json()['sentence']
#         vectorized_sentence = self.embedder.encode(sentence)
#         prediction_result = self.predictor.predict(vectorized_sentence)
#         return jsonify(prediction_result)

# if __name__ == '__main__':
#     app = Flask(__name__)
#     text_classifier = TextClassifier()
#     app.run(debug=True)

##-----------------------------------------------------------------------------

# from flask import Flask, request, jsonify
# from sentence_transformers import SentenceTransformer
# from joblib import load

# # Initialize the Flask app
# app = Flask(__name__)

# # Load models
# model_vectorize = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model_svc = load("trained_models/SVM.pkl")

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400

#     # Encode the input text
#     vectorized_text = model_vectorize.encode(data['text'])

#     # Predict using the SVM model
#     prediction = model_svc.predict([vectorized_text])

#     # Return the prediction as JSON
#     return jsonify({'prediction': prediction.tolist()})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

##-----------------------------------------------------------------------------

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from joblib import load

class TextClassifierAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.model_vectorize = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.model_svc = load("trained_models/SVM.pkl")
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json()
            if 'text' not in data:
                return jsonify({'error': 'No text provided'}), 400

            # Encode the input text
            vectorized_text = self.model_vectorize.encode(data['text'])

            # Predict using the SVM model
            prediction = self.model_svc.predict([vectorized_text])

            # Return the prediction as JSON
            return jsonify({'prediction': prediction.tolist()})

    def run(self, host='0.0.0.0', port=5000, debug=True):
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    api = TextClassifierAPI()
    api.run()
