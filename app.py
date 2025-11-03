import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Model Training and Loading ---
try:
    heart_data = pd.read_csv('heart.csv')
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please ensure it is in the same directory.")
    model = None

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict_heart_disease():
    if model is None:
        return jsonify({"error": "Model not loaded. Ensure 'heart.csv' is available."}), 500

    try:
        input_data = request.json
        input_list = [
            input_data['age'],
            input_data['sex'],
            input_data['cp'],
            input_data['trestbps'],
            input_data['chol'],
            input_data['fbs'],
            input_data['restecg'],
            input_data['thalach'],
            input_data['exang'],
            input_data['oldpeak'],
            input_data['slope'],
            input_data['ca'],
            input_data['thal']
        ]

        inp_data_as_numpy = np.asarray(input_list)
        inp_data_reshape = inp_data_as_numpy.reshape(1, -1)
        
        prediction = model.predict(inp_data_reshape)
        
        if prediction[0] == 0:
            result = "No Heart Disease Detected"
            prediction_value = 0
        else:
            result = "Heart Disease Detected"
            prediction_value = 1
        
        return jsonify({"prediction": prediction_value, "message": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)