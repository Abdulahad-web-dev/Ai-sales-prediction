import os
from datetime import datetime

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import get_products, load_assets, predict_sales


class PredictionRequest(BaseModel):
    product_description: str = Field(min_length=1, description="Exact product description from the training catalog")
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0, allow_inf_nan=False)
    timestamp: datetime = Field(description="Transaction date/time in ISO 8601 format")


class PredictionResponse(BaseModel):
    predicted_sales: float
    currency: str = "GBP"
    model: str


api = FastAPI(
    title="Retail Sales Prediction API",
    description="Predict transaction sales using the saved retail regression model.",
    version="1.0.0",
)


@api.get("/health")
def health_check():
    model, _ = load_assets()
    return {"status": "ok", "model": type(model).__name__}


@api.get("/api/products")
def products():
    return {"products": get_products()}


@api.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model, _ = load_assets()
        prediction = predict_sales(
            request.product_description,
            request.quantity,
            request.unit_price,
            request.timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(
        predicted_sales=prediction,
        currency="GBP",
        model=type(model).__name__,
    )


def predict_from_demo(product_description, quantity, unit_price, timestamp):
    try:
        prediction = predict_sales(product_description, int(quantity), unit_price, timestamp)
        return f"£{prediction:,.2f} GBP"
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc


model, _ = load_assets()
demo = gr.Interface(
    fn=predict_from_demo,
    inputs=[
        gr.Dropdown(choices=get_products(), value=get_products()[0], label="Product"),
        gr.Number(value=1, minimum=1, precision=0, label="Quantity"),
        gr.Number(value=10, minimum=0.01, label="Unit price (GBP)"),
        gr.DateTime(
            value=datetime(2011, 12, 9, 12, 0),
            include_time=True,
            type="datetime",
            label="Transaction date and time",
        ),
    ],
    outputs=gr.Textbox(label="Predicted transaction sales"),
    title="Retail Sales Predictor",
    description=f"Predictions from the saved {type(model).__name__} model. Choose a product found in the training catalog.",
    api_name=False,
)

app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))