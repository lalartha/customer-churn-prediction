import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and preprocessors
try:
    model = joblib.load('best_model_pipeline.pkl')
    scaler = joblib.load('scaler.pkl')
    le_geo = joblib.load('label_encoder_geography.pkl')
    le_gender = joblib.load('label_encoder_gender.pkl')
    feature_names = joblib.load('feature_names.pkl')
    logger.info("All model files loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Backend is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logger.info(f"Received prediction request: {data}")

        # Create DataFrame with proper feature encoding
        input_data = pd.DataFrame([{
            'CreditScore': float(data['CreditScore']),
            'Geography': le_geo.transform([data['Geography']])[0],
            'Gender': le_gender.transform([data['Gender']])[0],
            'Age': float(data['Age']),
            'Tenure': float(data['Tenure']),
            'Balance': float(data['Balance']),
            'NumOfProducts': float(data['NumOfProducts']),
            'HasCrCard': float(data['HasCrCard']),
            'IsActiveMember': float(data['IsActiveMember']),
            'EstimatedSalary': float(data['EstimatedSalary'])
        }])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])

        # Get feature importance (if available)
        xai_explanations = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(feature_names, importances, input_data.iloc[0].values),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            for feat_name, importance, feat_value in feature_importance:
                # Decode categorical features for display
                display_value = feat_value
                if feat_name == 'Geography':
                    display_value = le_geo.inverse_transform([int(feat_value)])[0]
                elif feat_name == 'Gender':
                    display_value = le_gender.inverse_transform([int(feat_value)])[0]

                xai_explanations.append({
                    'feature': feat_name,
                    'value': str(display_value),
                    'impact': float(importance)
                })

        response = {
            'prediction': prediction,
            'probability': probability,
            'xai_explanations': xai_explanations
        }

        logger.info(f"Prediction result: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Bank Churn Prediction API")
    print("=" * 60)
    print("Server running on: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("=" * 60)
    app.run(debug=True, port=5000, host='0.0.0.0')
