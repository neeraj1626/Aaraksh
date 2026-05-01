from flask import Blueprint, request, jsonify
from ml_integration.emergency_model_loader import check_emergency

emergency_bp = Blueprint('emergency_bp', __name__)

@emergency_bp.route('/check', methods=['POST'])
def detect_emergency():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    
    result = check_emergency(symptoms)
    return jsonify(result), 200