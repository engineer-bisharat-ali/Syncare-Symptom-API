from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("disease_predictor_model.pkl")
encoder = joblib.load("label_encoder.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

@app.route('/')
def home():
    return "✅ Symptom Prediction API is Running"

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_symptoms = data.get("symptoms", [])

    # Convert symptom list to feature vector
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptom_columns]
    prediction = model.predict([input_vector])[0]
    disease = encoder.inverse_transform([prediction])[0]

    return jsonify({"predicted_disease": disease})

# List all symptoms
@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify(symptom_columns)

# List all diseases
@app.route('/diseases', methods=['GET'])
def get_diseases():
    diseases = list(encoder.classes_)
    return jsonify(diseases)

if __name__ == '__main__':
    app.run(debug=True)
