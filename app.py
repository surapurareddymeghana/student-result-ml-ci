import time
import logging
import uuid
import joblib

from flask import Flask, request, jsonify, g, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# --------------------------------------------------
# 1. Flask Application
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# 2. Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# 3. Prometheus Metrics
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Model loaded status"
)

# --------------------------------------------------
# 4. Request Middleware
# --------------------------------------------------

@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):

    duration_seconds = time.perf_counter() - g.start_time
    duration_ms = duration_seconds * 1000

    if request.path != "/metrics":

        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_seconds)

    response.headers["X-Request-ID"] = g.request_id

    logger.info(
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response

# --------------------------------------------------
# 5. Load Machine Learning Model
# --------------------------------------------------

MODEL_PATH = "student_result_model.pkl"


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
        logger.info("event=model_loaded")
        return model

    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_loading_failed")
        raise

# --------------------------------------------------
# 6. Home Route
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Student Result Prediction API is running"
    })

# --------------------------------------------------
# 7. Health Check Route
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })

# --------------------------------------------------
# 8. Prediction Route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON input required"}), 400

        features = [[
            float(data["attendance"]),
            float(data["internal_marks"]),
            float(data["assignment_marks"]),
            float(data["previous_score"])
        ]]

        model = load_model()

        prediction = model.predict(features)

        result = int(prediction[0])

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id,
            result
        )

        return jsonify({
            "prediction": result,
            "result": "PASS" if result == 1 else "FAIL"
        })

    except Exception as error:

        PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": str(error)
        }), 500

# --------------------------------------------------
# 9. Prometheus Metrics Route
# --------------------------------------------------

@app.route("/metrics", methods=["GET"])
def metrics():

    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST
    )




# --------------------------------------------------
# 10. Start Application
# --------------------------------------------------

if __name__ == "__main__":

    logger.info("event=application_starting")

    app.run(
    host="0.0.0.0",
    port=5003
)