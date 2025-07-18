from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
df = pd.read_csv("mumbai-house-price-data-cleaned.csv")
localities = sorted(df['locality'].dropna().unique())
property_types = sorted(df['property_type'].dropna().unique())
cities = sorted(df['city'].dropna().unique())

# Load trained model pipeline
model = pickle.load(open("model.pkl", "rb"))

# Read the HTML file template with placeholders for form fields
with open("housing.html", "r", encoding="utf-8") as f:
    html_template = f.read()

@app.route("/")
def home():
    return render_template_string(
        html_template.replace("{{ prediction_text }}", ""), 
        localities=localities, 
        property_types=property_types,
        cities=cities,
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area_sqft"])
        locality = request.form["locality"]
        property_type = request.form["property_type"]
        city = request.form["city"]

        # Build a DataFrame with the same columns as your training DataFrame
        features = pd.DataFrame({
            "area": [area],
            "locality": [locality],
            "property_type": [property_type],
            "city": [city]
        })

        prediction = model.predict(features)
        price = round(prediction[0], 2)

        output_html = html_template.replace(
            "{{ prediction_text }}", f"<h3>Estimated House Price: ₹{price} Lakhs</h3>"
        )
        return render_template_string(output_html)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
