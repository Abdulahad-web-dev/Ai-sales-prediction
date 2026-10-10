from datetime import datetime
from functools import lru_cache
import math
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "best_model_xgb.pkl"
ENCODER_PATH = BASE_DIR / "outputs" / "label_encoder.pkl"
FEATURES = [
    "Description",
    "Quantity",
    "UnitPrice",
    "Month",
    "Day",
    "Year",
    "DayOfWeek",
    "Hour",
]


@lru_cache(maxsize=1)
def load_assets():
    if not MODEL_PATH.is_file() or not ENCODER_PATH.is_file():
        raise FileNotFoundError("Model files are missing from the outputs directory.")
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


def get_products():
    _, encoder = load_assets()
    return encoder.classes_.tolist()


def predict_sales(
    product_description: str,
    quantity: int,
    unit_price: float,
    timestamp: datetime,
) -> float:
    model, encoder = load_assets()

    if not isinstance(product_description, str) or not product_description.strip():
        raise ValueError("product_description must be a non-empty product from the training data.")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity must be a positive integer.")
    if isinstance(unit_price, bool) or not isinstance(unit_price, (int, float)):
        raise ValueError("unit_price must be a positive finite number.")
    if not math.isfinite(float(unit_price)) or unit_price <= 0:
        raise ValueError("unit_price must be a positive finite number.")

    try:
        encoded_product = int(encoder.transform([product_description])[0])
    except ValueError as exc:
        raise ValueError("product_description is not in the model's known product list.") from exc

    if timestamp is None or timestamp == "":
        raise ValueError("timestamp must be a valid date and time.")
    try:
        parsed_timestamp = pd.Timestamp(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be a valid date and time.") from exc
    if pd.isna(parsed_timestamp):
        raise ValueError("timestamp must be a valid date and time.")

    features = pd.DataFrame(
        [[
            encoded_product,
            quantity,
            float(unit_price),
            parsed_timestamp.month,
            parsed_timestamp.day,
            parsed_timestamp.year,
            parsed_timestamp.dayofweek,
            parsed_timestamp.hour,
        ]],
        columns=FEATURES,
    )
    return float(model.predict(features)[0])