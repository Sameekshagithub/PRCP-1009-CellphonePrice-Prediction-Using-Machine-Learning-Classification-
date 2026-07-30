"""
app.py
------
Flask web frontend for the Cellphone Price Prediction model.
Run this AFTER main.py has trained and saved a model (models/model_bundle.pkl
in the project root).

Usage:
    cd frontend
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model_bundle.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
needs_scaling = bundle["needs_scaling"]
model_name = bundle["model_name"]

PRICE_LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}

# Binary (Yes/No) specification fields — rendered as dropdowns in the form
BINARY_FIELDS = {"blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"}

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_columns=feature_columns,
        binary_fields=BINARY_FIELDS,
        model_name=model_name,
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {col: float(request.form.get(col, 0)) for col in feature_columns}
        input_df = pd.DataFrame([input_data], columns=feature_columns)
        input_arr = scaler.transform(input_df) if needs_scaling else input_df.values
        pred = int(model.predict(input_arr)[0])
        prediction = {"code": pred, "label": PRICE_LABELS.get(pred, "Unknown")}
    except Exception as e:
        prediction = {"error": str(e)}

    return render_template(
        "index.html",
        feature_columns=feature_columns,
        binary_fields=BINARY_FIELDS,
        model_name=model_name,
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)
