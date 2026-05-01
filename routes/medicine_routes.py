from flask import Blueprint, request, jsonify

medicine_bp = Blueprint('medicine_bp', __name__)

# Basic Knowledge Base
MEDICINE_DB = {
    "fever": ["Paracetamol", "Stay hydrated", "Rest"],
    "common cold": ["Antihistamines", "Warm fluids", "Vitamin C"],
    "high diabetes risk": ["Metformin (Consult Doctor)", "Low carb diet", "Daily 30min walk"],
    "high bp": ["Amlodipine (Consult Doctor)", "Low salt diet", "Stress management"]
}

@medicine_bp.route('/suggest', methods=['POST'])
def suggest_meds():
    data = request.get_json()
    condition = data.get('condition', '').lower()
    
    suggestions = MEDICINE_DB.get(condition, ["Please consult a specialist for this specific condition."])
    
    return jsonify({
        "condition": condition,
        "suggestions": suggestions,
        "disclaimer": "This is NOT a prescription. Consult a doctor before taking any medication."
    }), 200