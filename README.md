# House Price Predictor — Model Deployment

This week is the deployment part of the house price prediction project. The model was trained and tuned in Week 6 using the Ames Housing dataset, and here it is packaged into a small application that can actually take house details and return a predicted sale price.

There are two ways to use it:

- a Flask API for sending house information as JSON
- a Streamlit app with a simple form for entering the same information

The model is loaded from the saved `.pkl` file, so it does not retrain every time a prediction is made.

## How it works

```text
Week 6 notebook
      ↓
trained final model (LightGBM tuned)
      ↓
house_price_model_final.pkl
      ↓
utils/preprocessing.py
      ↓
Flask API ────────── Streamlit app
```

The preprocessing step is important because the new input needs to be prepared in the same way as the data used to train the Week 6 model. The saved model artifact contains the information needed for that, including the selected features, categorical/ordinal mappings, skewed-feature information, and the neighborhood tier mapping.

## Project structure

```text
week7-house-price-deployment/
├── model/
│   └── house_price_model_final.pkl
├── api/
│   ├── app.py
│   └── test_api.py
├── app/
│   └── streamlit_app.py
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## The model

The saved model is a tuned **LightGBM regressor**. It was trained to predict `log(SalePrice)`. When a prediction is made, the API converts that value back to the original dollar scale before returning it.

The model artifact also stores the 35 features used by the final model. The preprocessing code rebuilds the engineered features from the incoming house information, applies the transformations used in Week 6, creates the required dummy variables, and then selects the exact feature set expected by the model.

A `StandardScaler` is also stored in the artifact because it was part of the Week 6 preprocessing workflow. It is not applied to the final prediction, since the final model is LightGBM and the saved model expects the transformed feature values rather than scaled values.

## Setup

Clone the repository and move into the project folder:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd week7-house-price-deployment
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows, activate it with:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Before running anything, make sure this file is present:

```text
model/house_price_model_final.pkl
```

The API and Streamlit app both depend on it.

## Running the Flask API

Start the API with:

```bash
python api/app.py
```

The API runs locally at:

```text
http://127.0.0.1:5000
```

### Health check

In another terminal, run:

```bash
curl http://127.0.0.1:5000/health
```

A successful response looks like:

```json
{
    "status": "ok",
    "model": "LightGBM (tuned)",
    "number_of_features": 35
}
```

### Making a prediction

The `/predict` endpoint accepts a JSON object containing the house information.

For example:

```bash
curl -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/json"   -d '{
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
        "Neighborhood": "CollgCr",
        "MSZoning": "RL",
        "BsmtFinType2": "Unf",
        "GarageFinish": "RFn",
        "GarageType": "Attchd",
        "CentralAir": "Y",
        "ExterQual": "Gd",
        "ExterCond": "TA",
        "BsmtQual": "Gd",
        "KitchenQual": "Gd"
      }'
```

The API returns the predicted price in USD. A typical response has this format:

```json
{
    "model": "LightGBM (tuned)",
    "prediction": 207957.91,
    "currency": "USD"
}
```

Not every input field has to be supplied. The preprocessing code has default values for fields that are missing. If a value has the wrong type, such as text being supplied for a numeric field, the API returns a `400` response with an explanation instead of failing silently.

## Running the API tests

The tests are designed to be run while the Flask API is running.

In one terminal:

```bash
python api/app.py
```

Then, in another:

```bash
python api/test_api.py
```

The test script checks:

- the health endpoint
- a normal prediction
- rejection of invalid numeric input
- handling of an empty JSON object

## Running the Streamlit app

The Streamlit interface can be started with:

```bash
streamlit run app/streamlit_app.py
```

It opens locally at:

```text
http://localhost:8501
```

The app uses the same preprocessing and model as the Flask API. Instead of sending JSON manually, the user enters the house details through the form and clicks the prediction button. The resulting estimated price is displayed in the interface, along with the processed features used for the prediction.

## Limitations

The prediction is only an estimate based on the patterns in the Ames Housing data used to train the model. The training data covers homes in Ames, Iowa, from 2006–2010, so the model may not perform as well on houses that are substantially different from those examples.

This should therefore be treated as a machine-learning prediction rather than a professional property valuation or appraisal.

## Possible improvements

If the project were taken further, a few useful next steps would be:

- deploy the Flask API and Streamlit app so they can be accessed outside the local machine
- add authentication and rate limiting before making the API public
- log predictions and inputs so model performance can be monitored over time
- add more automated tests around the preprocessing and prediction pipeline

## Links

- GitHub repository: `https://github.com/0vSudo/week7-house-price-deployment`
- Deployed app/API: `https://week7-house-price-deployment-cbnbtcmuvcdbjrpgrxwu5t.streamlit.app/`
- Demo video: `https://drive.google.com/file/d/13qTpmFn8ajB_JF9-yeXY2g8ryLSIkrkD/view?usp=sharing`
- LinkedIn post: `https://www.linkedin.com/posts/0vsudo_machinelearning-python-datascience-share-7494901178812715008-FVKs/?utm_source=share&utm_medium=member_desktop&rcm=ACoAABTjX48B9flT98XklVubYhegY4XHBUkgdwo`
