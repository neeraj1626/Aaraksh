from flask import Blueprint, request, jsonify
from services.symptom_service import predict_symptom

symptom_bp = Blueprint("symptom", __name__)

@symptom_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No input provided"}), 400

        result = predict_symptom(data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500