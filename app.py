import os
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Base directory for relative paths (Required for Vercel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler using context managers
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Home route (for browser testing)
@app.route("/")
def home():
    return "Loan Prediction API is running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        # Validate required fields
        required_fields = [
            "Gender", "Married", "Dependents", "Education",
            "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
            "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Comprehensive value validation
        if data["Gender"] not in [0, 1]:
            return jsonify({"error": "Gender must be 0 (Female) or 1 (Male)"}), 400

        if data["Married"] not in [0, 1]:
            return jsonify({"error": "Married must be 0 (No) or 1 (Yes)"}), 400

        if data["Dependents"] not in [0, 1, 2, 3]:
            return jsonify({"error": "Dependents must be 0, 1, 2, or 3"}), 400

        if data["Education"] not in [0, 1]:
            return jsonify({"error": "Education must be 0 (Not Graduate) or 1 (Graduate)"}), 400

        if data["Self_Employed"] not in [0, 1]:
            return jsonify({"error": "Self_Employed must be 0 (No) or 1 (Yes)"}), 400

        if data["ApplicantIncome"] < 0:
            return jsonify({"error": "ApplicantIncome cannot be negative"}), 400

        if data["CoapplicantIncome"] < 0:
            return jsonify({"error": "CoapplicantIncome cannot be negative"}), 400

        if data["LoanAmount"] <= 0:
            return jsonify({"error": "LoanAmount must be positive"}), 400

        if data["Loan_Amount_Term"] <= 0:
            return jsonify({"error": "Loan_Amount_Term must be positive"}), 400

        if data["Credit_History"] not in [0, 1]:
            return jsonify({"error": "Credit_History must be 0 or 1"}), 400

        if data["Property_Area"] not in [0, 1, 2]:
            return jsonify({"error": "Property_Area must be 0 (Rural), 1 (Semiurban), or 2 (Urban)"}), 400

        # Convert input to array
        features = np.array([
            data["Gender"],
            data["Married"],
            data["Dependents"],
            data["Education"],
            data["Self_Employed"],
            data["ApplicantIncome"],
            data["CoapplicantIncome"],
            data["LoanAmount"],
            data["Loan_Amount_Term"],
            data["Credit_History"],
            data["Property_Area"]
        ]).reshape(1, -1)

        # Scale input
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]

        result = "Approved" if prediction == 1 else "Rejected"

        # Explanation logic
        reason = []

        if data["Credit_History"] == 1:
            reason.append("Good credit history")
        else:
            reason.append("Poor credit history")

        if data["ApplicantIncome"] > 4000:
            reason.append("Stable income")

        if data["LoanAmount"] > 200:
            reason.append("High loan amount")

        # Final response
        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "reason": ", ".join(reason)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)