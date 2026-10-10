---
title: Retail Sales Predictor
sdk: gradio
app_file: hf_app.py
python_version: "3.11"
---

# E-Commerce Sales Prediction

A Streamlit application that predicts the total sales value of a retail transaction from its quantity, unit price, product, and date/time features. The repository includes a trained model, an interactive prediction dashboard, a training and evaluation pipeline, and an exploratory notebook.

## Features

- Predict transaction sales using the saved regression model.
- Enter product, quantity, price, and date/time features in the dashboard.
- Optionally extract a price from a public product page URL.
- Review up to 10 recent predictions in the current Streamlit session and download them as CSV.
- Train and compare Linear Regression, Random Forest, and XGBoost models; save metrics and plots.
- Serve the saved model through a Hugging Face demo and a JSON API for AI Mart Pro.
- Inspect the included dashboard and model-result screenshots in `assets/`.

The range shown with each prediction is a simple plus/minus 8% estimate; it is not a statistically calibrated confidence interval.

## Screenshots

| Prediction dashboard | Model performance |
| --- | --- |
| ![Sales prediction dashboard](assets/dashboard_main.png) | ![Model performance charts](assets/model_performance.png) |

## Quick start

Requirements: Python 3.10 or later.

```powershell
git clone https://github.com/Abdulahad-web-dev/Ai-sales-prediction.git
cd Ai-sales-prediction
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>. The repository includes model files in `outputs/`, so you can launch the dashboard without first downloading the training dataset. Scikit-learn is pinned to 1.8.0 because that is the version used to serialize the included model artifacts.

On macOS or Linux, create and activate the environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then install the requirements and run `streamlit run app.py` as above.

## Hugging Face demo and API

The repository is configured as a Gradio Space. To publish it, create a **public Gradio Space** on Hugging Face and upload/push these files while preserving their paths:

```text
README.md
hf_app.py
inference.py
requirements.txt
outputs/best_model_xgb.pkl
outputs/label_encoder.pkl
```

The Space README metadata selects `hf_app.py` as its entry point. Once the Space is built, its public host is shown on the Space page. The browser demo is served at `/`, the health check at `/health`, and the prediction API at `/api/predict`. Interactive API documentation is available at `/docs`; `/api/products` returns the exact product descriptions accepted by the model.

Send `POST /api/predict` a JSON body with an exact product description from `/api/products`, a positive integer quantity, a positive unit price in GBP, and an ISO 8601 transaction timestamp:

```json
{
	"product_description": "WHITE HANGING HEART T-LIGHT HOLDER",
	"quantity": 2,
	"unit_price": 2.55,
	"timestamp": "2011-12-09T12:00:00"
}
```

Successful responses contain `predicted_sales`, `currency`, and `model`. Invalid values and products not in the training catalog return HTTP 422. The public Space has no API key by default; call it from the AI Mart Pro Express backend rather than exposing server credentials or depending on browser CORS.

Example Express-side request:

```js
const response = await fetch(`${process.env.SALES_PREDICTION_API_URL}/api/predict`, {
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify({
		product_description: "WHITE HANGING HEART T-LIGHT HOLDER",
		quantity: 2,
		unit_price: 2.55,
		timestamp: "2011-12-09T12:00:00",
	}),
});

if (!response.ok) {
	throw new Error(`Sales prediction failed: ${response.status}`);
}

const prediction = await response.json();
```

Set `SALES_PREDICTION_API_URL` to the Space host (for example, `https://<your-space-host>.hf.space`). Public free Spaces may sleep while idle and take time to wake up.

## Train the models

The training pipeline uses the UCI Online Retail dataset. Download and convert it to CSV before running the pipeline:

```powershell
python download_data.py
python convert_to_csv.py
python pipeline.py
```

The data is saved under `data/`, which is excluded from Git. The pipeline cleans the transactions, removes duplicates and invalid quantities/prices, creates `Sales` as `Quantity * UnitPrice`, extracts date/time features, label-encodes product descriptions, and samples up to 50,000 rows for training. It evaluates a random 80/20 train/test split and writes model artifacts and evaluation output under `outputs/`.

The original data source is the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), containing transactions from a UK-based online retailer.

## Evaluation snapshot

These metrics are from the included `outputs/results.csv`; they are a record of one training run, not a guarantee of future performance.

| Model | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.561 | 38.617 | 11.628 |
| Random Forest | 0.987 | 6.711 | 0.477 |
| XGBoost | 0.901 | 18.372 | 9.228 |

The included `outputs/best_model_xgb.pkl` is a legacy filename: inspecting the saved artifact shows it loads as a scikit-learn `GradientBoostingRegressor`, not an XGBoost estimator. It requires scikit-learn 1.8.0, pinned in `requirements.txt`. The `XGBoost` metrics row in `outputs/results.csv` records a training run and does not identify the estimator currently stored in that pickle. Metrics can vary when the dataset or training procedure changes.

## Repository layout

```text
app.py                       Streamlit prediction dashboard
hf_app.py                    Hugging Face Gradio demo and FastAPI API
inference.py                 Shared model loading and feature preparation
test_deployment.py           Hosted API route tests
pipeline.py                  Data preparation, training, and evaluation
download_data.py             Download the UCI Excel dataset
download_v2.py               Alternate dataset downloader
convert_to_csv.py            Convert the downloaded Excel file to CSV
requirements.txt              Python dependencies
notebooks/sales_prediction.ipynb
assets/                       Screenshots and generated charts
outputs/                      Model, encoder, metrics, and training plots
```

## Notes

- `outputs/best_model_xgb.pkl` and `outputs/label_encoder.pkl` are loaded by the app. If loading fails because of an environment or serialization incompatibility, the app attempts to train a fallback model on synthetic data.
- The optional product URL autofill depends on the target website being publicly reachable and exposing a price in markup the app recognizes; it may not work for every site.
- Training requires the UCI dataset and may take longer than starting the dashboard.
- The included `run_app.bat` references a machine-specific Python environment. Use `streamlit run app.py` from the activated environment for a portable launch.
