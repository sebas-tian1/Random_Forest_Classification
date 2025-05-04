from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

with open('test-model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        response = ''
        return response,200
    
    try:
        data = request.get_json()
        print("Data received: ", data)
        
        # Check for all missing features first
        missing_features = []
        for feature in feature_names:
            if feature not in data:
                missing_features.append(feature)
        
        # If any features are missing, return an error
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # All features present, continue with prediction
        input_data = {}
        for feature in feature_names:
            input_data[feature] = data[feature]
        
        X_pred = pd.DataFrame([input_data])
        print("Created DataFrame:", X_pred)
        X_pred = X_pred[feature_names]
        
        prediction = model.predict(X_pred)
        print("Prediction result:", prediction)

        return jsonify({'flowerType': prediction.tolist()[0]}), 200
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port= 5000, debug=True)