import firebase_admin
from firebase_admin import credentials, firestore
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(BASE_DIR, "..", "aaraksh-a2cd0-firebase-adminsdk-fbsvc-04578671ff.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ==========================
# GET CONTACTS
# ==========================
# ==========================
# GET USER DATA (NAME & CONTACTS)
# ==========================
def get_user_data(user_id):
    try:
        doc = db.collection("users").document(user_id).get()

        if doc.exists:
            data = doc.to_dict()
            # Returns a tuple: (list, string)
            return data.get("emergencyContacts", []), data.get("firstName", "User")

        return [], "User"
    except Exception as e:
        print("Firebase error:", e)
        return [], "User"
# ==========================
# SAVE CHAT
# ==========================
def save_chat(user_id, message, reply, risk):
    db.collection("users") \
      .document(user_id) \
      .collection("chats") \
      .add({
        "message": message,
        "reply": reply,
        "risk": risk,
        "timestamp": firestore.SERVER_TIMESTAMP
    })