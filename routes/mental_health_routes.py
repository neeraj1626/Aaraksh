from flask import Blueprint, request, jsonify
from services.mental_health_service import MentalHealthAssistant
from services.firebase_service import get_user_data

chat_bp = Blueprint("chat", __name__)

assistant = MentalHealthAssistant()

@chat_bp.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        # ==========================
        # VALIDATION
        # ==========================
        if not data:
            return jsonify({"error": "No data provided"}), 400

        message = data.get("message")
        user_id = data.get("user_id")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # ==========================
        # FETCH CONTACTS
        # ==========================
        try:
            contacts, first_name = get_user_data(user_id)
        except Exception as e:
            print("Firebase Contact Error:", e)
            contacts, first_name = [], "User"

        # ==========================
        # PROCESS CHAT
        # ==========================
        result = assistant.process_chat(message, user_id)

        # Attach contacts for frontend
        result["contacts"] = contacts
        result["firstName"] = first_name

        return jsonify(result), 200

    except Exception as e:
        print("ROUTE ERROR:", e)
        return jsonify({
            "error": "Internal Server Error",
            "details": str(e)
        }), 500
        
@chat_bp.route("/chat/history/<user_id>", methods=["GET"])
def get_chat_history(user_id):
    try:
        from services.firebase_service import db
        from firebase_admin import firestore

        chats_ref = db.collection("users") \
                      .document(user_id) \
                      .collection("chats") \
                      .order_by("timestamp", direction=firestore.Query.DESCENDING) \
                      .limit(50)

        chats = []

        for doc in chats_ref.stream():
            data = doc.to_dict()
            chats.append({
                "message": data.get("message"),
                "reply": data.get("reply"),
                "risk": data.get("risk"),
                "timestamp": str(data.get("timestamp"))
            })

        return jsonify({"chats": chats}), 200

    except Exception as e:
        print("HISTORY ERROR:", e)
        return jsonify({"error": str(e)}), 500