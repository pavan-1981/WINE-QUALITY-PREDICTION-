from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__, static_folder='.')

# Global variable to hold our trained model
model = None

def train_model():
    global model
    print("Loading dataset and training the model...")
    # Load dataset
    df = pd.read_csv('winequality-red.csv')
    
    # Features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Model trained successfully!")

# Serve the main index.html
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Serve other static files (HTML, CSS, JS)
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join('.', path)):
        return send_from_directory('.', path)
    return "File not found", 404

# Prediction API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract features in the correct order corresponding to the CSV columns
    # fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
    # free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
    try:
        features = [[
            float(data['fixed_acidity']),
            float(data['volatile_acidity']),
            float(data['citric_acid']),
            float(data['residual_sugar']),
            float(data['chlorides']),
            float(data['free_so2']),
            float(data['total_so2']),
            float(data['density']),
            float(data['ph']),
            float(data['sulphates']),
            float(data['alcohol'])
        ]]
        
        # Predict
        prediction = model.predict(features)[0]
        # Get probability (confidence)
        probas = model.predict_proba(features)[0]
        confidence = max(probas) * 100
        
        return jsonify({
            'score': float(prediction),
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    train_model()
    # Run the server on port 8000
    app.run(host='0.0.0.0', port=8000, debug=True)
