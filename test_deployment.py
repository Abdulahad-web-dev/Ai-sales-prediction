import unittest

from fastapi.testclient import TestClient

from hf_app import app
from inference import get_products


class DeploymentApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.product = get_products()[0]

    def test_health_reports_loaded_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_gradio_demo_is_mounted_at_root(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Retail Sales Predictor", response.text)

    def test_products_returns_known_catalog(self):
        response = self.client.get("/api/products")

        self.assertEqual(response.status_code, 200)
        self.assertIn(self.product, response.json()["products"])

    def test_predict_returns_sales_estimate(self):
        response = self.client.post(
            "/api/predict",
            json={
                "product_description": self.product,
                "quantity": 1,
                "unit_price": 10.0,
                "timestamp": "2011-12-09T12:00:00",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["predicted_sales"], 0)
        self.assertEqual(response.json()["currency"], "GBP")

    def test_predict_rejects_unknown_product(self):
        response = self.client.post(
            "/api/predict",
            json={
                "product_description": "not in training catalog",
                "quantity": 1,
                "unit_price": 10.0,
                "timestamp": "2011-12-09T12:00:00",
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()