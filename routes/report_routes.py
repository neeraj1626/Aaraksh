from flask import Blueprint, request, jsonify
from services.report_service import process_report

report_bp = Blueprint("report", __name__)

@report_bp.route("/analyze-report", methods=["POST"])
def analyze_report():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        result = process_report(file)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500