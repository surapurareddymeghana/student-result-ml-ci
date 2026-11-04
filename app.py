
import logging
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, g
import joblib
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    if not MODEL_PATH.exists():
        logger.error("event=model_load_failed model_file=%s", MODEL_PATH)
        raise FileNotFoundError("Student result model not found.")

    model = joblib.load(MODEL_PATH)
    logger.info("event=model_loaded model_file=%s", MODEL_PATH)
    return model


# Log every HTTP request
@app.before_request
def before_request():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    duration = (time.perf_counter() - g.start_time) * 1000

    level = logging.WARNING if response.status_code >= 400 else logging.INFO

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration
    )

    response.headers["X-Request-ID"] = g.request_id
    return response


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning(
            "event=invalid_request request_id=%s reason=missing_json_body",
            g.request_id
        )
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error request_id=%s missing_fields=%s",
            g.request_id,
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature] for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id,
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    logger.info("event=application_starting")
    app.run(host="0.0.0.0", port=5000)
