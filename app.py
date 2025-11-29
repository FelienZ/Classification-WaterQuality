from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'https://el-prediksi.vercel.app/ml'])

# Load model
with open("water_quality_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # JSON dari frontend
    X_input = pd.DataFrame([data])
    
    # Impute sini fitur extra
    extra_cols = ["Total Dissolved Solids", "Manganese", "Iron"]
    for col in extra_cols:
        if col not in X_input.columns:
            X_input[col] = None
    X_input = X_input.reindex(model.feature_names_in_, axis=1)
    prediction = model.predict(X_input)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

