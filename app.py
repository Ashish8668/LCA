from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("API_KEY")
genai.configure(api_key=gemini_api_key)

# Flask app
app = Flask(__name__)

# Load saved model, scaler, encoders
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))

# Function to safely encode unseen categories
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat_gemini", methods=["POST"])
def chat_gemini():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "Please enter a valid question."})

    try:
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Generate response
        response = model.generate_content(query)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Input values from form
        Material = request.form["Material"]
        Source = float(request.form["Source_%"])
        Quantity = float(request.form["Quantity_tons"])
        Ore_Grade = float(request.form["Ore_Grade_%"])
        Energy = float(request.form["Energy_%"])
        TransportMode = request.form["Transport_Mode"]
        Distance = float(request.form["Distance_km"])
        SmeltingEff = float(request.form["Smelting_Efficiency_%"])

        # Encode categorical features safely
        Material = safe_encode(encoders["Metal"], Material)
        TransportMode = safe_encode(encoders["Transport_Mode"], TransportMode)

        # Feature vector in same order as training
        features = np.array([[Material, Source, Quantity, Ore_Grade, Energy,
                              TransportMode, Distance, SmeltingEff]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        preds = model.predict(features_scaled)[0]

        output = {
            "Carbon_Footprint_kgCO2": round(preds[0], 2),
            "Water_Use_m3": round(preds[1], 2),
            "Energy_Intensity_MJ": round(preds[2], 2),
            "Land_Disturbance_m2": round(preds[3], 2),
            "Reuse_%": round(preds[4], 2),
            "Recycle_%": round(preds[5], 2),
            "Global_Warming_Potential": round(preds[6], 2),
            "End_of_Life_Score": round(preds[7], 2)
        }

        return render_template("index.html", prediction=output)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
