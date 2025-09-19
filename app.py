from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io 
import base64 

import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
import seaborn as sns 
sns.set_theme(style="whitegrid") 

# Configure Gemini API
load_dotenv()
gemini_api_key = os.getenv("API_KEY")
genai.configure(api_key=gemini_api_key)

# Flask app
app = Flask(__name__)

# Load saved model, scaler, encoders
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))

# Safe encoder function
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,      # no predictions yet
        recommendations=None, # no Gemini output yet
        charts={},            # empty dict for charts
        inputs={}             # empty dict for input form
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


# ---------- Helper to convert plot to base64 ----------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close(fig)
    return data

def create_base64_plot(fig):
    """Convert Matplotlib fig to base64 string for embedding in HTML"""
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close(fig)
    return plot_url

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Input values
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

        # ML Prediction
        # preds = model.predict(features_scaled)[0]
        # output = {
        #     "Carbon_Footprint_kgCO2": round(preds[0], 2),
        #     "Water_Use_m3": round(preds[1], 2),
        #     "Energy_Intensity_MJ": round(preds[2], 2),
        #     "Land_Disturbance_m2": round(preds[3], 2),
        #     "Reuse_%": round(preds[4], 2),
        #     "Recycle_%": round(preds[5], 2),
        #     "Global_Warming_Potential": round(preds[6], 2),
        #     "End_of_Life_Score": round(preds[7], 2)
        # }

        if model is None:
            # fallback/static output
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
            # ML Prediction
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
        
        charts = {}

        # 1. Bar Chart
        fig, ax = plt.subplots()
        sns.barplot(
            x=["Carbon", "Water", "Energy", "Land"],
            y=[
                output["Carbon_Footprint_kgCO2"],
                output["Water_Use_m3"],
                output["Energy_Intensity_MJ"],
                output["Land_Disturbance_m2"]
            ],
            ax=ax
        )
        ax.set_title("Impact Distribution")
        charts["bar"] = create_base64_plot(fig)

        # 2. Pie Chart (Reuse vs Recycle vs End of Life)
        fig, ax = plt.subplots()
        values = [output["Reuse_%"], output["Recycle_%"], output["End_of_Life_Score"] * 100]
        labels = ["Reuse %", "Recycle %", "End of Life %"]
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title("Circular Economy Breakdown")
        charts["pie"] = create_base64_plot(fig)

        # 3. Line Chart (trend of impacts)
        fig, ax = plt.subplots()
        impacts = ["Carbon", "Water", "Energy", "Land"]
        values = [
            output["Carbon_Footprint_kgCO2"],
            output["Water_Use_m3"],
            output["Energy_Intensity_MJ"],
            output["Land_Disturbance_m2"]
        ]
        ax.plot(impacts, values, marker="o")
        ax.set_title("Impact Trend")
        charts["line"] = create_base64_plot(fig)

        # 4. Radar Chart (Spider Plot)
        categories = ["Carbon", "Water", "Energy", "Land", "Reuse", "Recycle"]
        values = [
            output["Carbon_Footprint_kgCO2"] / 10,  # scaled for visualization
            output["Water_Use_m3"],
            output["Energy_Intensity_MJ"] / 20,
            output["Land_Disturbance_m2"] / 5,
            output["Reuse_%"],
            output["Recycle_%"]
        ]
        values += values[:1]  # close loop
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title("Radar Chart of Impacts")
        charts["radar"] = create_base64_plot(fig)

        # 5. Stacked Bar Chart
        fig, ax = plt.subplots()
        reuse = output["Reuse_%"]
        recycle = output["Recycle_%"]
        waste = 100 - reuse - recycle
        ax.bar("Lifecycle", reuse, label="Reuse %")
        ax.bar("Lifecycle", recycle, bottom=reuse, label="Recycle %")
        ax.bar("Lifecycle", waste, bottom=reuse+recycle, label="Waste %")
        ax.legend()
        ax.set_title("Reuse vs Recycle vs Waste")
        charts["stacked"] = create_base64_plot(fig)

        # 6. Heatmap (correlation of impacts)
        data = pd.DataFrame({
            "Carbon": [output["Carbon_Footprint_kgCO2"]],
            "Water": [output["Water_Use_m3"]],
            "Energy": [output["Energy_Intensity_MJ"]],
            "Land": [output["Land_Disturbance_m2"]],
            "Reuse": [output["Reuse_%"]],
            "Recycle": [output["Recycle_%"]],
            "EndLife": [output["End_of_Life_Score"]]
        })
        corr = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        ax.set_title("Impact Correlation Heatmap")
        charts["heatmap"] = create_base64_plot(fig)

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

        inputs = request.form
        return render_template("index.html", 
                               prediction=output, recommendations=recommendations,
                               charts=charts,
                               inputs=inputs)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
