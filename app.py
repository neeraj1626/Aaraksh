from flask import Flask
from routes.symptom_routes import symptom_bp
from routes.report_routes import report_bp
from routes.mental_health_routes import chat_bp

app = Flask(__name__)

# Register route
app.register_blueprint(symptom_bp)
app.register_blueprint(report_bp)
app.register_blueprint(chat_bp)

@app.route("/")
def home():
    return "AI Health Guardian API Running"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)