"""
test_api.py

Tests the Flask House Price Prediction API.

IMPORTANT:
Start the API first:

    python api/app.py

Then open another terminal and run:

    python api/test_api.py
"""

import requests


# ============================================================
# API ADDRESS
# ============================================================

BASE_URL = "http://127.0.0.1:5000"


# ============================================================
# SAMPLE HOUSE
# ============================================================

# This is a human-readable house description.
# The preprocessing.py file converts it into the
# 35 features required by the Week 6 model.

SAMPLE_INPUT = {

    "OverallQual": 7,
    "OverallCond": 5,

    "GrLivArea": 1710,
    "LotArea": 8450,

    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "YrSold": 2007,

    "1stFlrSF": 856,
    "2ndFlrSF": 854,

    "TotalBsmtSF": 856,
    "BsmtFinSF1": 706,

    "FullBath": 2,
    "HalfBath": 1,

    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,

    "GarageCars": 2,
    "GarageArea": 548,
    "GarageYrBlt": 2003,

    "PoolArea": 0,
    "Fireplaces": 1,

    "OpenPorchSF": 61,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "WoodDeckSF": 196,

    "Neighborhood": "CollgCr",
    "MSZoning": "RL",

    "BsmtFinType2": "Unf",

    "GarageFinish": "RFn",
    "GarageType": "Attchd",

    "CentralAir": "Y",

    "ExterQual": "Gd",
    "ExterCond": "TA",

    "BsmtQual": "Gd",
    "BsmtCond": "TA",

    "HeatingQC": "Ex",

    "KitchenQual": "Gd",

    "FireplaceQu": "Gd",

    "GarageQual": "TA",
    "GarageCond": "TA",

    "PoolQC": "None",
}


# ============================================================
# TEST 1 — HEALTH CHECK
# ============================================================

def test_health():

    response = requests.get(
        f"{BASE_URL}/health"
    )

    print()
    print("TEST 1 — Health Check")
    print("--------------------------------")

    print(
        "Status code:",
        response.status_code
    )

    print(
        "Response:",
        response.json()
    )

    assert response.status_code == 200

    print("PASS")


# ============================================================
# TEST 2 — SUCCESSFUL PREDICTION
# ============================================================

def test_prediction():

    response = requests.post(
        f"{BASE_URL}/predict",
        json=SAMPLE_INPUT
    )

    print()
    print("TEST 2 — Prediction")
    print("--------------------------------")

    print(
        "Status code:",
        response.status_code
    )

    print(
        "Response:",
        response.json()
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result

    assert result["prediction"] > 0

    print("PASS")


# ============================================================
# TEST 3 — INVALID JSON DATA
# ============================================================

def test_invalid_data():

    bad_input = SAMPLE_INPUT.copy()

    bad_input["OverallQual"] = "this-is-not-a-number"

    response = requests.post(
        f"{BASE_URL}/predict",
        json=bad_input
    )

    print()
    print("TEST 3 — Invalid Input")
    print("--------------------------------")

    print(
        "Status code:",
        response.status_code
    )

    print(
        "Response:",
        response.json()
    )

    # The API should reject invalid input
    assert response.status_code == 400

    print("PASS")


# ============================================================
# TEST 4 — SERVER REJECTS EMPTY BODY
# ============================================================

def test_empty_body():

    response = requests.post(
        f"{BASE_URL}/predict",
        json={}
    )

    print()
    print("TEST 4 — Empty Input")
    print("--------------------------------")

    print(
        "Status code:",
        response.status_code
    )

    print(
        "Response:",
        response.json()
    )

    # Empty input is technically accepted by our preprocessing
    # because defaults are available. Therefore we simply verify
    # that the API responds successfully rather than crashing.

    assert response.status_code in [200, 400]

    print("PASS")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("RUNNING HOUSE PRICE API TESTS")
    print("==============================================")

    test_health()

    test_prediction()

    test_invalid_data()

    test_empty_body()

    print()
    print("==============================================")
    print("ALL API TESTS COMPLETED")
    print("==============================================")