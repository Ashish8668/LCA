from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from dotenv import load_dotenv
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import google.generativeai as genai
import json
import plotly.graph_objects as go

# Configure Gemini API
load_dotenv()
gemini_api_key = os.getenv("API_KEY")
genai.configure(api_key=gemini_api_key)

app = Flask(__name__)

# Load model, scaler, encoders
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))

def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

def create_base64_plot(fig):
    """Convert Matplotlib figure to base64"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close(fig)
    return plot_url

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        recommendations=None,
        charts={},
        sankeys={}
    )

@app.route("/chat_gemini", methods=["POST"])
def chat_gemini():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"answer": "Please enter a valid question."})
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(query)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Inputs
        Material = request.form["Material"]
        Source = float(request.form["Source_%"])
        Quantity = float(request.form["Quantity_tons"])
        Ore_Grade = float(request.form["Ore_Grade_%"])
        Energy = float(request.form["Energy_%"])
        TransportMode = request.form["Transport_Mode"]
        Distance = float(request.form["Distance_km"])
        SmeltingEff = float(request.form["Smelting_Efficiency_%"])

        # Encode categorical
        Material_enc = safe_encode(encoders["Metal"], Material)
        Transport_enc = safe_encode(encoders["Transport_Mode"], TransportMode)

        # Features
        features = np.array([[Material_enc, Source, Quantity, Ore_Grade, Energy,
                              Transport_enc, Distance, SmeltingEff]])
        features_scaled = scaler.transform(features)

        # Prediction
        if model is None:
            output = {
                "Carbon_Footprint_kgCO2": 120.0,
                "Water_Use_m3": 30.0,
                "Energy_Intensity_MJ": 500.0,
                "Land_Disturbance_m2": 200.0,
                "Reuse_%": 60.0,
                "Recycle_%": 75.0,
                "Global_Warming_Potential": 1.8,
                "End_of_Life_Score": 90.0
            }
        else:
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

        # ---------- Charts ----------
        charts = {}

        # Radar Chart in %
        max_vals = {
            "Carbon_Footprint_kgCO2": 500,
            "Water_Use_m3": 100,
            "Energy_Intensity_MJ": 2000,
            "Land_Disturbance_m2": 500,
            "Reuse_%": 100,
            "Recycle_%": 100,
            "End_of_Life_Score": 100
        }
        categories = ["Carbon", "Water", "Energy", "Land", "Reuse", "Recycle", "End of Life"]
        values = [
            output["Carbon_Footprint_kgCO2"]/max_vals["Carbon_Footprint_kgCO2"]*100,
            output["Water_Use_m3"]/max_vals["Water_Use_m3"]*100,
            output["Energy_Intensity_MJ"]/max_vals["Energy_Intensity_MJ"]*100,
            output["Land_Disturbance_m2"]/max_vals["Land_Disturbance_m2"]*100,
            output["Reuse_%"]/max_vals["Reuse_%"]*100,
            output["Recycle_%"]/max_vals["Recycle_%"]*100,
            output["End_of_Life_Score"]/max_vals["End_of_Life_Score"]*100
        ]
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title("Radar Chart of Impacts (%)")
        charts["radar"] = create_base64_plot(fig)

        # Pie Chart
        fig, ax = plt.subplots()
        reuse = output["Reuse_%"]
        recycle = output["Recycle_%"]
        waste = 100 - reuse - recycle
        ax.pie([reuse, recycle, waste], labels=["Reuse %", "Recycle %", "Waste %"],
               autopct='%1.1f%%', colors=["#4CAF50","#2196F3","#F44336"])
        ax.set_title("Circularity Breakdown")
        charts["pie_waste"] = create_base64_plot(fig)

        # Sankey JSON
        sankeys = {}
        material_labels = ["Metal Source", "Processing", "Transport", "Product", "Reuse", "Recycle", "Waste"]
        material_sources = [0,1,2,3,3,3]
        material_targets = [1,2,3,4,5,6]
        material_values = [Quantity, Quantity, Quantity, reuse, recycle, waste]

        energy_labels = ["Grid Mix", "Processing", "Carbon", "Water", "Energy", "Land"]
        energy_sources = [0,1,1,1,1]
        energy_targets = [1,2,3,4,5]
        energy_values = [
            Energy,
            output["Carbon_Footprint_kgCO2"],
            output["Water_Use_m3"],
            output["Energy_Intensity_MJ"],
            output["Land_Disturbance_m2"]
        ]
        sankeys["material"] = {
            "labels": material_labels,
            "sources": material_sources,
            "targets": material_targets,
            "values": material_values
        }

        sankeys["energy"] = {
            "labels": energy_labels,
            "sources": energy_sources,
            "targets": energy_targets,
            "values": energy_values
        }

        # Gemini Recommendations
        try:
            context = f"""
            Inputs:
            Material = {Material}, Source% = {Source}, Quantity = {Quantity} tons,
            Ore Grade = {Ore_Grade}%, Energy = {Energy}%, Transport Mode = {TransportMode}, Distance = {Distance} km,
            Smelting Efficiency = {SmeltingEff}%

            Outputs (ML Predictions):
            {output}

            Task: Give exactly 2 practical recommendations to improve circularity across full value chain.
            """
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            rec_response = model_gemini.generate_content(context)
            recommendations = rec_response.text.strip()
        except Exception as e:
            recommendations = f"Could not fetch recommendations: {str(e)}"

        return render_template(
            "index.html",
            prediction=output,
            recommendations=recommendations,
            charts=charts,
            sankeys=sankeys,
            inputs=request.form
        )

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
