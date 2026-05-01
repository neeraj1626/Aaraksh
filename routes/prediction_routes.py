from flask import Blueprint, request, jsonify

prediction_bp = Blueprint('prediction_bp', __name__)

@prediction_bp.route('/risk', methods=['POST'])
def predict_risk():
    data = request.get_json()
    
    # Logic: Basic rule-based risk for now
    age = data.get('age', 0)
    bmi = data.get('bmi', 0)
    
    risk_level = "Low"
    if age > 45 and bmi > 30:
        risk_level = "High"
    elif age > 30 or bmi > 25:
        risk_level = "Moderate"

    return jsonify({
        "risk_level": risk_level,
        "message": f"Based on your profile, your health risk is {risk_level}."
    }), 200