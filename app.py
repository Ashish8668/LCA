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
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from flask import send_file

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

def create_base64_plot(fig, **kwargs):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', **kwargs)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close(fig)
    return plot_url

@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    try:
        data = request.get_json()

        inputs = data.get("inputs")
        linear = data.get("linear")
        circular = data.get("circular")
        charts = data.get("charts", {})  # base64 images
        sankeys = data.get("sankeys", {}) 
        recommendations = data.get("recommendations", "No recommendations available")

        pdf_filename = "circularity_report.pdf"
        create_pdf_report(
            filename=pdf_filename,
            inputs=inputs,
            linear=linear,
            circular=circular,
            charts=charts,
            sankeys=sankeys,
            recommendations=recommendations
        )

        return send_file(pdf_filename, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)})

def plotly_sankey_to_base64(sankey_data):
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data["labels"],
            color="lightblue"
        ),
        link=dict(
            source=sankey_data["sources"],
            target=sankey_data["targets"],
            value=sankey_data["values"],
            color=sankey_data.get("colors", None)
        )
    )])
    fig.update_layout(font_size=10)

    try:
        # Preferred: uses Kaleido under the hood
        img_bytes = fig.to_image(format="png")
    except Exception as e:
        # Fallback: try writing to a temp file then read
        try:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.write_image(tmp.name, format="png")
            tmp.close()
            with open(tmp.name, "rb") as f:
                img_bytes = f.read()
            # optionally remove the temp file
            try:
                os.remove(tmp.name)
            except Exception:
                pass
        except Exception as e2:
            raise RuntimeError(
                "Failed to export Plotly figure to image. "
                "Make sure 'kaleido' is installed (pip install kaleido). "
                f"Original error: {e}; fallback error: {e2}"
            )

    return base64.b64encode(img_bytes).decode("utf8")

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        recommendations=None,
        charts={},
        sankeys={}
    )

def create_pdf_report(filename, inputs, linear, circular, charts, sankeys , recommendations):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    elements.append(Paragraph("Circularity Cockpit – LCA Report", styles['Title']))
    elements.append(Spacer(1,12))
    elements.append(Paragraph(f"Material: {inputs['Material']}, Quantity: {inputs['Quantity_tons']} tons", styles['Normal']))
    elements.append(Spacer(1,12))
    
    # 1. Input Table
    elements.append(Paragraph("1. Input Summary", styles['Heading2']))
    input_data = [["Parameter","Value"]] + [[k,v] for k,v in inputs.items()]
    table = Table(input_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER')
    ]))
    elements.append(table)
    elements.append(Spacer(1,12))
    
    # 2. Circularity Indicators
    elements.append(Paragraph("2. Circularity Indicators (Linear vs Circular)", styles['Heading2']))
    headers = ["Indicator","Linear","Circular","Unit"]
    data = [headers]
    units = {
        "Carbon_Footprint_kgCO2":"kgCO₂",
        "Water_Use_m3":"m³",
        "Energy_Intensity_MJ":"MJ",
        "Land_Disturbance_m2":"m²",
        "Reuse_%":"%",
        "Recycle_%":"%",
        "Global_Warming_Potential":"kgCO₂-eq",
        "End_of_Life_Score":"%"
    }
    for key in linear.keys():
        row = [
            key,
            linear[key],
            circular[key],
            units.get(key,"")
        ]
        data.append(row)
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER')
    ]))
    elements.append(table)
    elements.append(Spacer(1,12))

    # 3b. Sankey Diagrams
    if sankeys:   # assuming sankeys is passed separately as a dict
        elements.append(Paragraph("Sankey Diagrams", styles['Heading2']))
        for skey, sankey_data in sankeys.items():
            elements.append(Paragraph(skey.replace("_", " ").title(), styles['Heading3']))
            sankey_b64 = plotly_sankey_to_base64(sankey_data)
            img_bytes = base64.b64decode(sankey_b64)
            img_buffer = io.BytesIO(img_bytes)
            img = Image(img_buffer, width=400, height=300)
            elements.append(img)
            elements.append(Spacer(1, 12))
    else:
        elements.append(Paragraph("No Sankey diagrams available.", styles['Normal']))

    # 3. Visualizations
    elements.append(Paragraph("3. Visualizations", styles['Heading2']))
    for chart_key in charts:
        elements.append(Paragraph(chart_key.replace("_"," ").title(), styles['Heading3']))
        # Decode base64 image
        img_bytes = base64.b64decode(charts[chart_key])
        img_buffer = io.BytesIO(img_bytes)
        img = Image(img_buffer, width=400, height=300)
        elements.append(img)
        elements.append(Spacer(1,12))
    
    # 4. Recommendations
    elements.append(Paragraph("4. AI Recommendations", styles['Heading2']))
    elements.append(Paragraph(recommendations, styles['Normal']))
    
    # Build PDF
    doc.build(elements)


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
    

def model_predict(features_scaled):
    """Return a structured output dict from the model (or fallback)."""
    if model is None:
        return {
            "Carbon_Footprint_kgCO2": 120.0,
            "Water_Use_m3": 30.0,
            "Energy_Intensity_MJ": 500.0,
            "Land_Disturbance_m2": 200.0,
            "Reuse_%": 60.0,
            "Recycle_%": 20.0,
            "Production_Cost": 1000.0,
            "Global_Warming_Potential": 1.8,
            "End_of_Life_Score": 90.0
        }
    preds = model.predict(features_scaled)[0]
    # map predictions to fields (adjust indices if your model differs)
    out = {
        "Carbon_Footprint_kgCO2": float(round(preds[0],2)),
        "Water_Use_m3": float(round(preds[1],2)),
        "Energy_Intensity_MJ": float(round(preds[2],2)),
        "Land_Disturbance_m2": float(round(preds[3],2)),
        "Reuse_%": float(round(preds[4],2)),
        "Recycle_%": float(round(preds[5],2)),
        "Global_Warming_Potential": float(round(preds[6] if len(preds)>6 else 0.0,2)),
        "End_of_Life_Score": float(round(preds[7] if len(preds)>7 else 50.0,2))
    }
    return out


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

        output = model_predict(features_scaled)

        # ---- Derive two scenarios from base_out ----
        # Linear (baseline) : lower reuse/recycle, higher impacts
        linear = dict(output)  # shallow copy
        linear["Reuse_%"] = min(linear.get("Reuse_%", 5), 10) if linear.get("Reuse_%") is not None else 5.0
        linear["Recycle_%"] = min(linear.get("Recycle_%", 5), 10) if linear.get("Recycle_%") is not None else 5.0
        linear_scale = 1.25  # worsen impacts
        linear["Carbon_Footprint_kgCO2"] = round(linear["Carbon_Footprint_kgCO2"] * linear_scale, 2)
        linear["Water_Use_m3"] = round(linear["Water_Use_m3"] * linear_scale, 2)
        linear["Energy_Intensity_MJ"] = round(linear["Energy_Intensity_MJ"] * linear_scale, 2)
        linear["Land_Disturbance_m2"] = round(linear["Land_Disturbance_m2"] * linear_scale, 2)
        linear["End_of_Life_Score"] = max(5.0, round(linear.get("End_of_Life_Score",50)*0.7,2))

        # Circular (improved) : higher reuse/recycle, lower impacts
        circular = dict(output)
        circular["Reuse_%"] = max(circular.get("Reuse_%",50), 50)
        circular["Recycle_%"] = max(circular.get("Recycle_%",65), 65)
        circular_scale = 0.6  # improve impacts
        circular["Carbon_Footprint_kgCO2"] = round(circular["Carbon_Footprint_kgCO2"] * circular_scale, 2)
        circular["Water_Use_m3"] = round(circular["Water_Use_m3"] * circular_scale, 2)
        circular["Energy_Intensity_MJ"] = round(circular["Energy_Intensity_MJ"] * circular_scale, 2)
        circular["Land_Disturbance_m2"] = round(circular["Land_Disturbance_m2"] * circular_scale, 2)
        circular["End_of_Life_Score"] = min(100.0, round(circular.get("End_of_Life_Score",50)*1.1,2))

        # ensure reuse+recycle <= 100
        for s in (linear, circular):
            total = s["Reuse_%"] + s["Recycle_%"]
            if total > 100:
                s["Reuse_%"] = round(s["Reuse_%"]/total*100,2)
                s["Recycle_%"] = round(s["Recycle_%"]/total*100,2)

        # ---------- Charts ----------
        charts = {}

        # Radar Chart in %
        # Radar Chart - scale each metric independently
        categories = ["Carbon","Water","Energy","Land","Reuse","Recycle","End of Life"]
        values_raw = [
            output["Carbon_Footprint_kgCO2"],
            output["Water_Use_m3"],
            output["Energy_Intensity_MJ"],
            output["Land_Disturbance_m2"],
            output["Reuse_%"],
            output["Recycle_%"],
            output["End_of_Life_Score"]
        ]

        # scale each value to 0-100 using linear mapping: value/max(value for this metric)
        # Here we assume output itself is representative
        # If comparing Linear vs Circular, use max of both for that metric
        values = []
        for i, val in enumerate(values_raw):
            # max scaling per metric (here just one scenario)
            scaled = val / max(values_raw[i],1) * 100  # avoid zero-division
            values.append(scaled)

        # close radar
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4,3), subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 100)
        ax.set_title("Radar Chart of Impacts (%)")
        charts["radar"] = create_base64_plot(fig)

        # Pie Chart
        fig, ax = plt.subplots(figsize=(4,3))
        reuse = output["Reuse_%"]
        recycle = output["Recycle_%"]
        waste = max(0.0, 100 - reuse - recycle)
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

        # Grouped Bar (Carbon, Water, Energy, Land)
        metrics = ["Carbon_Footprint_kgCO2","Water_Use_m3","Energy_Intensity_MJ","Land_Disturbance_m2"]
        left_vals = [linear[m] for m in metrics]
        right_vals = [circular[m] for m in metrics]
        labels = ["Carbon","Water","Energy","Land"]

        fig, ax = plt.subplots(figsize=(4,3))
        ind = np.arange(len(labels))
        width = 0.35
        ax.bar(ind - width/2, left_vals, width, label="Linear")
        ax.bar(ind + width/2, right_vals, width, label="Circular")
        ax.set_xticks(ind); ax.set_xticklabels(labels)
        ax.set_title("Impact Comparison (Linear vs Circular)")
        ax.legend()
        charts["grouped_bar"] = create_base64_plot(fig, dpi=120)

        # Radar Chart - normalize per metric max of the two
        categories = ["Carbon","Water","Energy","Land","Reuse","Recycle","End of Life"]
        left_raw = [
            linear["Carbon_Footprint_kgCO2"],
            linear["Water_Use_m3"],
            linear["Energy_Intensity_MJ"],
            linear["Land_Disturbance_m2"],
            linear["Reuse_%"],
            linear["Recycle_%"],
            linear["End_of_Life_Score"]
        ]
        right_raw = [
            circular["Carbon_Footprint_kgCO2"],
            circular["Water_Use_m3"],
            circular["Energy_Intensity_MJ"],
            circular["Land_Disturbance_m2"],
            circular["Reuse_%"],
            circular["Recycle_%"],
            circular["End_of_Life_Score"]
        ]
        combined_max = [max(l,r,1.0) for l,r in zip(left_raw,right_raw)]
        left_pct = [ (l/m)*100 for l,m in zip(left_raw,combined_max) ]
        right_pct = [ (r/m)*100 for r,m in zip(right_raw,combined_max) ]

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        left_plot = left_pct + left_pct[:1]
        right_plot = right_pct + right_pct[:1]

        fig, ax = plt.subplots(figsize=(4,3), subplot_kw=dict(polar=True))
        ax.plot(angles, left_plot, 'o-', linewidth=2, label='Linear')
        ax.fill(angles, left_plot, alpha=0.12)
        ax.plot(angles, right_plot, 'o-', linewidth=2, label='Circular')
        ax.fill(angles, right_plot, alpha=0.12)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0,100)
        ax.set_title("Normalized Radar: Linear vs Circular (%)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.25,1.15))
        charts["radar_compare"] = create_base64_plot(fig, dpi=120)

        # Pie charts (end-of-life)
        def pie_b64(reuse, recycle):
            waste = max(0.0, 100.0 - reuse - recycle)
            fig, ax = plt.subplots(figsize=(4,3))
            ax.pie([reuse, recycle, waste], labels=["Reuse %","Recycle %","Waste %"], autopct='%1.1f%%', colors=["#4CAF50","#2196F3","#F44336"])
            ax.set_title("End-of-life breakdown")
            return create_base64_plot(fig, dpi=120)

        charts["pie_linear"] = pie_b64(linear["Reuse_%"], linear["Recycle_%"])
        charts["pie_circular"] = pie_b64(circular["Reuse_%"], circular["Recycle_%"])

        # Sankey: create combined sankey with duplicated links for Linear (red) and Circular (blue)
        # Node indices: 0:Source,1:Processing,2:Transport,3:Product,4:Reuse,5:Recycle,6:Waste
        labels = ["Source","Processing","Transport","Product","Reuse","Recycle","Waste"]

        # compute mass flows using Quantity (tons) * percent/100 for reuse/recycle/waste
        linear_reuse_mass = Quantity * linear["Reuse_%"] / 100.0
        linear_recycle_mass = Quantity * linear["Recycle_%"] / 100.0
        linear_waste_mass = max(0.0, Quantity - linear_reuse_mass - linear_recycle_mass)

        circular_reuse_mass = Quantity * circular["Reuse_%"] / 100.0
        circular_recycle_mass = Quantity * circular["Recycle_%"] / 100.0
        circular_waste_mass = max(0.0, Quantity - circular_reuse_mass - circular_recycle_mass)

        # links: for both scenarios: Source->Processing, Processing->Transport, Transport->Product, Product->Reuse/Recycle/Waste
        # We'll duplicate the chain for linear and circular but they must target same nodes; link values differ.
        sources = []
        targets = []
        values = []
        colors = []

        # Linear chain (use red-ish)
        linear_color = "rgba(228, 26, 28, 0.6)"
        chain_indices = [(0,1, Quantity),(1,2, Quantity),(2,3, Quantity),
                         (3,4, linear_reuse_mass),(3,5, linear_recycle_mass),(3,6, linear_waste_mass)]
        for s,t,v in chain_indices:
            sources.append(s); targets.append(t); values.append(round(v,3)); colors.append(linear_color)

        # Circular chain (use blue-ish) -- to visually overlay, use smaller alpha
        circ_color = "rgba(31, 119, 180, 0.6)"
        chain_indices_c = [(0,1, Quantity),(1,2, Quantity),(2,3, Quantity),
                           (3,4, circular_reuse_mass),(3,5, circular_recycle_mass),(3,6, circular_waste_mass)]
        for s,t,v in chain_indices_c:
            sources.append(s); targets.append(t); values.append(round(v,3)); colors.append(circ_color)

        sankeys = {
            "material": {
                "labels": labels,
                "sources": sources,
                "targets": targets,
                "values": values,
                "colors": colors,
                "legend": {"linear_color": linear_color, "circular_color": circ_color}
            }
        }

        # Energy sankey: we create two sets of links from Grid->Processing->Impacts with different values
        # Use energy intensity * Quantity as a proxy for total energy
        left_energy_total = linear["Energy_Intensity_MJ"] * Quantity
        right_energy_total = circular["Energy_Intensity_MJ"] * Quantity
        e_labels = ["Grid Mix","Processing","Carbon","Water","Energy","Land"]
        e_sources = []
        e_targets = []
        e_values = []
        e_colors = []

        # linear links
        e_chain_lin = [(0,1,left_energy_total),(1,2,linear["Carbon_Footprint_kgCO2"]),(1,3,linear["Water_Use_m3"]),
                       (1,4,linear["Energy_Intensity_MJ"]),(1,5,linear["Land_Disturbance_m2"])]
        for s,t,v in e_chain_lin:
            e_sources.append(s); e_targets.append(t); e_values.append(round(v,3)); e_colors.append(linear_color)

        # circular
        e_chain_circ = [(0,1,right_energy_total),(1,2,circular["Carbon_Footprint_kgCO2"]),(1,3,circular["Water_Use_m3"]),
                        (1,4,circular["Energy_Intensity_MJ"]),(1,5,circular["Land_Disturbance_m2"])]
        for s,t,v in e_chain_circ:
            e_sources.append(s); e_targets.append(t); e_values.append(round(v,3)); e_colors.append(circ_color)

        sankeys["energy"] = {
            "labels": e_labels,
            "sources": e_sources,
            "targets": e_targets,
            "values": e_values,
            "colors": e_colors,
            "legend": {"linear_color": linear_color, "circular_color": circ_color}
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

        comparisons = {
            "linear": {
                "Carbon": linear["Carbon_Footprint_kgCO2"],
                "Water": linear["Water_Use_m3"],
                "Energy": linear["Energy_Intensity_MJ"],
                "Land": linear["Land_Disturbance_m2"],
                "Reuse": linear["Reuse_%"],
                "Recycle": linear["Recycle_%"],
                "End_of_Life": linear["End_of_Life_Score"]
            },
            "circular": {
                "Carbon": circular["Carbon_Footprint_kgCO2"],
                "Water": circular["Water_Use_m3"],
                "Energy": circular["Energy_Intensity_MJ"],
                "Land": circular["Land_Disturbance_m2"],
                "Reuse": circular["Reuse_%"],
                "Recycle": circular["Recycle_%"],
                "End_of_Life": circular["End_of_Life_Score"]
            }
        }

        return render_template(
            "index.html",
            prediction=output,
            recommendations=recommendations,
            charts=charts,
            sankeys=sankeys,
            linear=linear,
            circular=circular,
            comparisons=comparisons,
            charts_meta={"grouped_bar":"grouped_bar","radar":"radar_compare"},
            inputs=request.form,
            title="Comparative Dashboard"
        )

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
