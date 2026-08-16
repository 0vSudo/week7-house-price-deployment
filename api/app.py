"""
app.py

Flask REST API for the Week 6 House Price Prediction model.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
from flask import Flask, jsonify, request


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PREPROCESSING
# ============================================================

from utils.preprocessing import (
    preprocess_input
)


# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "house_price_model_final.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    artifact = joblib.load(
        MODEL_PATH
    )

    model = artifact["model"]

    MODEL_NAME = artifact.get(
        "final_model_key",
        "Week 6 House Price Model"
    )

    SELECTED_FEATURES = artifact.get(
        "selected_features",
        []
    )

    print(
        "=============================================="
    )

    print(
        "House Price Prediction API"
    )

    print(
        "=============================================="
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Features: {len(SELECTED_FEATURES)}"
    )

    print(
        f"Model file: {MODEL_PATH}"
    )

    print(
        "Model loaded successfully."
    )

    print(
        "=============================================="
    )


except FileNotFoundError:

    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}\n\n"
        "Make sure house_price_model_final.pkl "
        "is inside the model folder."
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status": "ok",
            "model": MODEL_NAME,
            "number_of_features": len(
                SELECTED_FEATURES
            ),
        }
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Get JSON
    # --------------------------------------------------------

    payload = request.get_json(
        silent=True
    )

    if payload is None:

        return jsonify(
            {
                "error":
                    "Request body must contain "
                    "valid JSON."
            }
        ), 400

    # --------------------------------------------------------
    # Validate JSON object
    # --------------------------------------------------------

    if not isinstance(
        payload,
        dict
    ):

        return jsonify(
            {
                "error":
                    "JSON body must be an object."
            }
        ), 400

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    try:

        X = preprocess_input(
            payload,
            artifact
        )

        prediction_log = model.predict(
            X
        )[0]

        prediction = np.expm1(
            prediction_log
        )

        prediction = max(
            0,
            float(prediction)
        )

        return jsonify(
            {
                "model": MODEL_NAME,
                "prediction": round(
                    prediction,
                    2
                ),
                "currency": "USD",
            }
        )

    except Exception as error:

        return jsonify(
            {
                "error":
                    "Prediction failed.",
                "details":
                    str(error),
            }
        ), 400


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )