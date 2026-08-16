"""
preprocessing.py

Prepares new house information in the same format used by the
Week 6 house price prediction model.
"""

import numpy as np
import pandas as pd


# ============================================================
# DEFAULT VALUES
# ============================================================

DEFAULT_VALUES = {
    "OverallQual": 6,
    "OverallCond": 5,
    "GrLivArea": 1500,
    "LotArea": 8000,
    "YearBuilt": 2000,
    "YearRemodAdd": 2000,
    "YrSold": 2008,

    "1stFlrSF": 1000,
    "2ndFlrSF": 500,
    "TotalBsmtSF": 800,
    "BsmtFinSF1": 400,

    "FullBath": 2,
    "HalfBath": 1,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,

    "GarageCars": 2,
    "GarageArea": 500,
    "GarageYrBlt": 2000,

    "PoolArea": 0,
    "Fireplaces": 1,

    "OpenPorchSF": 50,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "WoodDeckSF": 50,

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
# NUMERIC INPUT COLUMNS
# ============================================================

NUMERIC_INPUT_COLUMNS = {
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "YrSold",
    "1stFlrSF",
    "2ndFlrSF",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
    "GarageYrBlt",
    "PoolArea",
    "Fireplaces",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "WoodDeckSF",
}


# ============================================================
# FALLBACK QUALITY MAPPING
# ============================================================

FALLBACK_QUALITY_MAP = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
}


FALLBACK_ORDINAL_COLUMNS = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(
    df,
    neighborhood_tier_map=None
):
    """
    Recreate the engineered features used in Week 6.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Total square footage
    # --------------------------------------------------------

    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    # --------------------------------------------------------
    # Total bathrooms
    # --------------------------------------------------------

    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # --------------------------------------------------------
    # Total porch/deck area
    # --------------------------------------------------------

    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    # --------------------------------------------------------
    # House age
    # --------------------------------------------------------

    df["HouseAge"] = (
        df["YrSold"]
        - df["YearBuilt"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # Remodeling age
    # --------------------------------------------------------

    df["RemodAge"] = (
        df["YrSold"]
        - df["YearRemodAdd"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # Quality x living area interaction
    # --------------------------------------------------------

    df["OverallQual_x_GrLivArea"] = (
        df["OverallQual"]
        * df["GrLivArea"]
    )

    # --------------------------------------------------------
    # Binary features
    # --------------------------------------------------------

    df["HasPool"] = (
        df["PoolArea"] > 0
    ).astype(int)

    df["HasGarage"] = (
        df["GarageArea"] > 0
    ).astype(int)

    df["HasFireplace"] = (
        df["Fireplaces"] > 0
    ).astype(int)

    df["Has2ndFlr"] = (
        df["2ndFlrSF"] > 0
    ).astype(int)

    df["HasBsmt"] = (
        df["TotalBsmtSF"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Neighborhood tier
    # --------------------------------------------------------

    if neighborhood_tier_map is not None:

        df["NeighborhoodTier"] = (
            df["Neighborhood"]
            .map(neighborhood_tier_map)
            .fillna("Mid")
        )

    else:

        df["NeighborhoodTier"] = "Mid"

    return df


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_input(
    input_data,
    artifact
):
    """
    Convert raw house information into the exact feature
    format expected by the Week 6 model.
    """

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(input_data, dict):

        df = pd.DataFrame(
            [input_data]
        )

    elif isinstance(input_data, pd.DataFrame):

        df = input_data.copy()

    else:

        raise TypeError(
            "input_data must be a dictionary "
            "or pandas DataFrame."
        )

    # --------------------------------------------------------
    # Validate numeric inputs
    # --------------------------------------------------------

    for column in NUMERIC_INPUT_COLUMNS:

        if column in df.columns:

            value = df[column].iloc[0]

            if pd.isna(value):
                continue

            try:

                float(value)

            except (TypeError, ValueError):

                raise ValueError(
                    f"{column} must be a number."
                )

    # --------------------------------------------------------
    # Add missing fields
    # --------------------------------------------------------

    for column, default_value in DEFAULT_VALUES.items():

        if column not in df.columns:

            df[column] = default_value

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

    for column in NUMERIC_INPUT_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise"
            )

    # --------------------------------------------------------
    # Fill missing categorical values
    # --------------------------------------------------------

    object_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in object_columns:

        df[column] = df[column].fillna(
            "None"
        )

    # --------------------------------------------------------
    # Fill missing numeric values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        df[column] = df[column].fillna(0)

    # --------------------------------------------------------
    # Recreate Week 6 engineered features
    # --------------------------------------------------------

    neighborhood_map = artifact.get(
        "tier_map",
        {}
    )

    df = create_engineered_features(
        df,
        neighborhood_map
    )

    # --------------------------------------------------------
    # Apply Week 6 skewed-feature transformations
    # --------------------------------------------------------

    skewed_features = artifact.get(
        "skewed_features",
        []
    )

    for column in skewed_features:

        if column in df.columns:

            df[column] = np.log1p(
                df[column].clip(lower=0)
            )

    # --------------------------------------------------------
    # Apply Week 6 ordinal encoding
    # --------------------------------------------------------

    quality_map = artifact.get(
        "qual_map",
        FALLBACK_QUALITY_MAP
    )

    ordinal_columns = artifact.get(
        "ordinal_cols",
        FALLBACK_ORDINAL_COLUMNS
    )

    for column in ordinal_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .map(quality_map)
                .fillna(0)
            )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if categorical_columns:

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=False,
            dtype=float
        )

    # --------------------------------------------------------
    # Get the feature columns used during Week 6
    # --------------------------------------------------------

    scaler = artifact.get(
        "scaler"
    )

    if scaler is None:

        raise ValueError(
            "The saved model artifact does not contain "
            "the expected scaler information."
        )

    all_feature_columns = list(
        scaler.feature_names_in_
    )

    # --------------------------------------------------------
    # Recreate the complete Week 6 feature matrix
    # --------------------------------------------------------

    df = df.reindex(
        columns=all_feature_columns,
        fill_value=0
    )

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.fillna(0)

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # The final LightGBM model was trained using the
    # transformed feature values, not the StandardScaler
    # output.
    #
    # The scaler is retained in the Week 6 artifact because
    # it was part of the original preprocessing workflow,
    # but it is NOT applied to the final LightGBM prediction.
    # --------------------------------------------------------

    selected_features = artifact.get(
        "selected_features"
    )

    if not selected_features:

        raise ValueError(
            "The saved model artifact does not contain "
            "selected_features."
        )

    # --------------------------------------------------------
    # Return the exact 35 features required by the model
    # --------------------------------------------------------

    return df.reindex(
        columns=selected_features,
        fill_value=0
    )