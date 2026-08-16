"""
streamlit_app.py

Streamlit web application for the Week 6 House Price
Prediction model.

Run from the project root:

    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys

# Allow imports from the project root
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

import joblib
import streamlit as st

# Import Week 6 preprocessing
from utils.preprocessing import preprocess_input


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "house_price_model_final.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    artifact = joblib.load(
        MODEL_PATH
    )

    return artifact


try:

    artifact = load_model()

    model = artifact["model"]

    MODEL_NAME = artifact.get(
        "final_model_key",
        "House Price Model"
    )

except FileNotFoundError:

    st.error(
        "Model file not found. "
        "Please place house_price_model_final.pkl "
        "inside the model folder."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏠 House Price Predictor")

st.write(
    "Enter the characteristics of a house below and "
    "the Week 6 machine learning model will estimate "
    "its sale price."
)

st.caption(
    f"Model used: {MODEL_NAME}"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("About this project")

st.sidebar.write(
    """
This application is the Week 7 deployment stage of my
House Price Prediction project.

The machine learning model was developed during Week 6
using feature engineering, feature selection and model
optimization.

The final model is served here through a Streamlit
interface.
"""
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("house_prediction_form"):

    st.subheader("🏡 Property Information")

    # --------------------------------------------------------
    # Basic property characteristics
    # --------------------------------------------------------

    st.markdown("### Basic Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:

        overall_qual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=6,
            help="Overall material and finish quality."
        )

    with col2:

        overall_cond = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=5,
        )

    with col3:

        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=500,
            max_value=100000,
            value=8000,
            step=100,
        )

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    st.markdown("### House Size")

    col1, col2, col3 = st.columns(3)

    with col1:

        gr_liv_area = st.number_input(
            "Living Area (sq ft)",
            min_value=200,
            max_value=10000,
            value=1500,
            step=50,
        )

    with col2:

        first_floor = st.number_input(
            "1st Floor Area (sq ft)",
            min_value=0,
            max_value=10000,
            value=1000,
            step=50,
        )

    with col3:

        second_floor = st.number_input(
            "2nd Floor Area (sq ft)",
            min_value=0,
            max_value=10000,
            value=500,
            step=50,
        )

    # --------------------------------------------------------
    # Basement
    # --------------------------------------------------------

    st.markdown("### Basement")

    col1, col2, col3 = st.columns(3)

    with col1:

        basement_area = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=800,
            step=50,
        )

    with col2:

        basement_finished = st.number_input(
            "Finished Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=400,
            step=50,
        )

    with col3:

        basement_quality = st.selectbox(
            "Basement Quality",
            [
                "None",
                "Po",
                "Fa",
                "TA",
                "Gd",
                "Ex",
            ],
            index=4,
        )

    # --------------------------------------------------------
    # Bathrooms
    # --------------------------------------------------------

    st.markdown("### Bathrooms")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        full_bath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=6,
            value=2,
            step=1,
        )

    with col2:

        half_bath = st.number_input(
            "Half Bathrooms",
            min_value=0,
            max_value=4,
            value=1,
            step=1,
        )

    with col3:

        basement_full_bath = st.number_input(
            "Basement Full Bathrooms",
            min_value=0,
            max_value=4,
            value=1,
            step=1,
        )

    with col4:

        basement_half_bath = st.number_input(
            "Basement Half Bathrooms",
            min_value=0,
            max_value=4,
            value=0,
            step=1,
        )

    # --------------------------------------------------------
    # Garage
    # --------------------------------------------------------

    st.markdown("### Garage")

    col1, col2, col3 = st.columns(3)

    with col1:

        garage_cars = st.number_input(
            "Garage Capacity",
            min_value=0,
            max_value=6,
            value=2,
            step=1,
        )

    with col2:

        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            max_value=3000,
            value=500,
            step=25,
        )

    with col3:

        garage_year = st.number_input(
            "Garage Year Built",
            min_value=0,
            max_value=2026,
            value=2000,
            step=1,
        )

    # --------------------------------------------------------
    # House age
    # --------------------------------------------------------

    st.markdown("### Dates")

    col1, col2, col3 = st.columns(3)

    with col1:

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=2000,
            step=1,
        )

    with col2:

        year_remod = st.number_input(
            "Year Remodeled",
            min_value=1800,
            max_value=2026,
            value=2000,
            step=1,
        )

    with col3:

        year_sold = st.number_input(
            "Year Sold",
            min_value=2006,
            max_value=2026,
            value=2008,
            step=1,
        )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    st.markdown("### Location")

    col1, col2 = st.columns(2)

    with col1:

        zoning = st.selectbox(
            "MS Zoning",
            [
                "RL",
                "RM",
                "FV",
            ],
            index=0,
        )

    with col2:

        neighborhood = st.selectbox(
            "Neighborhood",
            [
                "CollgCr",
                "Veenker",
                "Crawfor",
                "NoRidge",
                "Mitchel",
                "Somerst",
                "NWAmes",
                "OldTown",
                "BrkSide",
                "Sawyer",
                "NridgHt",
                "NAmes",
                "SawyerW",
                "IDOTRR",
                "MeadowV",
                "Edwards",
                "Timber",
                "Gilbert",
                "ClearCr",
                "StoneBr",
                "NPkVill",
                "Blmngtn",
                "BrDale",
                "SWISU",
                "Blueste",
            ],
            index=0,
        )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    st.markdown("### Quality")

    col1, col2, col3 = st.columns(3)

    quality_options = [
        "None",
        "Po",
        "Fa",
        "TA",
        "Gd",
        "Ex",
    ]

    with col1:

        exterior_quality = st.selectbox(
            "Exterior Quality",
            quality_options,
            index=4,
        )

    with col2:

        kitchen_quality = st.selectbox(
            "Kitchen Quality",
            quality_options,
            index=4,
        )

    with col3:

        exterior_condition = st.selectbox(
            "Exterior Condition",
            quality_options,
            index=3,
        )

    # --------------------------------------------------------
    # Other characteristics
    # --------------------------------------------------------

    st.markdown("### Other Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        central_air = st.selectbox(
            "Central Air",
            ["Y", "N"],
            index=0,
        )

    with col2:

        pool_area = st.number_input(
            "Pool Area (sq ft)",
            min_value=0,
            max_value=3000,
            value=0,
            step=50,
        )

    with col3:

        fireplaces = st.number_input(
            "Number of Fireplaces",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
        )

    # --------------------------------------------------------
    # Submit
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "🔮 Predict House Price",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Build raw input dictionary
    # --------------------------------------------------------

    house_data = {

        "OverallQual": overall_qual,
        "OverallCond": overall_cond,

        "GrLivArea": gr_liv_area,
        "LotArea": lot_area,

        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "YrSold": year_sold,

        "1stFlrSF": first_floor,
        "2ndFlrSF": second_floor,

        "TotalBsmtSF": basement_area,
        "BsmtFinSF1": basement_finished,

        "FullBath": full_bath,
        "HalfBath": half_bath,

        "BsmtFullBath": basement_full_bath,
        "BsmtHalfBath": basement_half_bath,

        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "GarageYrBlt": garage_year,

        "PoolArea": pool_area,
        "Fireplaces": fireplaces,

        # Defaults for engineered features
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "WoodDeckSF": 50,

        # Categorical variables
        "Neighborhood": neighborhood,
        "MSZoning": zoning,

        "BsmtFinType2": "Unf",

        "GarageFinish": "RFn",
        "GarageType": "Attchd",

        "CentralAir": central_air,

        "ExterQual": exterior_quality,
        "ExterCond": exterior_condition,

        "BsmtQual": basement_quality,
        "BsmtCond": "TA",

        "HeatingQC": "Ex",

        "KitchenQual": kitchen_quality,

        "FireplaceQu": "Gd",

        "GarageQual": "TA",
        "GarageCond": "TA",

        "PoolQC": "None",
    }

    # --------------------------------------------------------
    # Run preprocessing
    # --------------------------------------------------------

    try:

        X = preprocess_input(
            house_data,
            artifact
        )

        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction_log = model.predict(X)[0]

        # Model predicts log(SalePrice), so convert it back
        # to the original dollar scale.
        import numpy as np

        prediction = np.expm1(
            prediction_log
        )

        prediction = max(
            0,
            float(prediction)
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.success(
            "Prediction generated successfully!"
        )

        st.metric(
            label="Estimated Sale Price",
            value=f"${prediction:,.0f}",
        )

        st.info(
            "This is a machine-learning estimate, "
            "not a professional property valuation."
        )

        # ----------------------------------------------------
        # Show processed features
        # ----------------------------------------------------

        with st.expander(
            "View model input features"
        ):

            st.dataframe(
                X,
                use_container_width=True
            )

    except Exception as error:

        st.error(
            "Something went wrong while generating "
            "the prediction."
        )

        st.exception(error)