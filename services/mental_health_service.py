import os
import time
import google.generativeai as genai
from datetime import datetime
from collections import defaultdict
from enum import Enum
from services.firebase_service import get_user_data, save_chat
# ============================================================
# GEMINI CONFIGURATION
# ============================================================
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="""You are a professional, empathetic Mental Health Assistant.
    Provide short, supportive responses.
    Never give medical advice.
    Keep responses under 80 words."""
)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class MentalHealthAssistant:
    def __init__(self):
        self.sessions = defaultdict(lambda: {
            "messages": [],
            "summary": "Initial conversation started."
        })
        self.last_call_time = 0
        self.last_sos_time = {}  # ⏱️ prevent repeated SOS

    # ========================================================
    # QUICK LOCAL RESPONSES (NO API)
    # ========================================================
    def quick_reply(self, message):
        msg = message.lower()

        if any(w in msg for w in ["hi", "hello", "hey"]):
            return "Hey, I’m here for you. How are you feeling today?"

        if any(w in msg for w in ["thanks", "thank you"]):
            return "You're always welcome. I'm here for you."

        return None

    # ========================================================
    # SAFE RISK DETECTION
    # ========================================================
    def detect_risk(self, message):
        msg = message.lower()

        critical_keywords = [
            "kill myself", "suicide", "end my life",
            "i want to die", "i will die", "no reason to live"
        ]

        high_keywords = [
            "hopeless", "worthless", "can't go on", "depressed"
        ]

        medium_keywords = [
            "sad", "stressed", "anxious", "tired"
        ]

        if any(w in msg for w in critical_keywords):
            return "CRITICAL"

        if any(w in msg for w in high_keywords):
            return "HIGH"

        if any(w in msg for w in medium_keywords):
            return "MEDIUM"

        return "LOW"

    # ========================================================
    # CONTEXT BUILDER
    # ========================================================
    def build_context(self, user_id):
        history = self.sessions[user_id]["messages"][-4:]
        context = ""

        for m in history:
            role = "User" if m["is_user"] else "Assistant"
            context += f"{role}: {m['text']}\n"

        return context

    # ========================================================
    # MAIN CHAT FUNCTION
    # ========================================================
    def process_chat(self, message, user_id):
        print("🔥 GEMINI CALLED")
        print("API KEY:", API_KEY)
        try:
            if not message.strip():
                return {"reply": "I'm listening. Tell me what's on your mind."}

            # ==========================
            # STEP 1: QUICK LOCAL REPLY
            # ==========================
            local = self.quick_reply(message)
            if local:
                return {
                    "reply": local,
                    "risk_level": "LOW",
                    "trigger_sos": False
                }

            # ==========================
            # STEP 2: RISK DETECTION
            # ==========================
            risk_lvl = self.detect_risk(message)

            # ==========================
            # 🚨 CRITICAL HANDLING (STEALTH SOS)
            # ==========================
            if risk_lvl == "CRITICAL":
                current_time = time.time()

                # Prevent repeated SOS within 5 minutes
                if user_id in self.last_sos_time:
                    if current_time - self.last_sos_time[user_id] < 300:
                        return {
                            "reply": "I'm really concerned about you. I'm here with you. Tell me more.",
                            "risk_level": "CRITICAL",
                            "trigger_sos": False
                        }

                self.last_sos_time[user_id] = current_time

                try:
                    contacts, first_name = get_user_data(user_id)
                except:
                    contacts, first_name = [], "User"

                # Save emergency event
                try:
                    save_chat(user_id, message, "CRITICAL ALERT TRIGGERED", "CRITICAL")
                except Exception as e:
                    print("Firebase Error:", e)

                return {
                    "reply": "I’m really concerned about you. You don’t have to go through this alone. I'm here with you.",
                    "risk_level": "CRITICAL",
                    "trigger_sos": False,
                    "contacts": contacts,
                    "firstName": first_name
                }

            # ==========================
            # STEP 3: COOLDOWN (FREE TIER SAFE)
            # ==========================
            current_time = time.time()

            if current_time - self.last_call_time < 3:
                return {
                    "reply": "I'm here with you. Tell me more…",
                    "risk_level": risk_lvl,
                    "trigger_sos": False
                }

            self.last_call_time = current_time

            # ==========================
            # STEP 4: GEMINI CALL
            # ==========================
            context = self.build_context(user_id)

            try:
                response = model.generate_content(f"""
                Context:
                {context}

                User: "{message}"

                Respond empathetically in under 80 words.
                """)

                reply = response.text.strip()

            except Exception as e:
                print("AI Error:", e)

                if "429" in str(e):
                    return {
                        "reply": "I'm still here with you. Let’s take it slow… tell me more.",
                        "risk_level": risk_lvl,
                        "trigger_sos": False
                    }

                reply = "I'm here for you. Want to share more?"

            # ==========================
            # MEMORY
            # ==========================
            self.sessions[user_id]["messages"].append({"text": message, "is_user": True})
            self.sessions[user_id]["messages"].append({"text": reply, "is_user": False})

            # ==========================
            # SAVE CHAT
            # ==========================
            try:
                save_chat(user_id, message, reply, risk_lvl)
            except Exception as e:
                print("Firebase Error:", e)

            return {
                "reply": reply,
                "risk_level": risk_lvl,
                "trigger_sos": False,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print("SERVER ERROR:", e)
            return {
                "reply": "Something went wrong. Please try again.",
                "risk_level": "LOW",
                "trigger_sos": False
            }
