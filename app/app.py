"""
Flask Web API for Credit Card Fraud Detection
Provides a web interface to make fraud predictions
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src to path so we can import predictor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from redictor import FraudPredictor

# Initialize Flask app
app = Flask(__name__)

# Initialize predictor (load model and scaler once at startup)
print("Loading fraud detection model...")
try:
    predictor = FraudPredictor(
        model_path='models/random_forest.pkl',
        scaler_path='models/scaler.pkl'
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None


@app.route('/')
def home():
    """Home page - display prediction form"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for fraud prediction
    Receives transaction data as JSON
    Returns prediction result as JSON
    """
    
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the server.'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate data
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Make prediction
        result = predictor.predict_single(data)
        
        # Return result as JSON
        return jsonify({
            'success': True,
            'fraud': result['fraud'],
            'probability': round(result['probability'], 4),
            'confidence': round(result['confidence'], 2),
            'message': f"Transaction flagged as {result['fraud']} with {result['confidence']:.2f}% confidence"
        })
    
    except Exception as e:
        # Return error message
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/health')
def health():
    """Health check endpoint - verify API is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'Fraud detection API is running'
    })


@app.route('/about')
def about():
    """About page - project information"""
    return jsonify({
        'project': 'Credit Card Fraud Detection',
        'model': 'Random Forest Classifier',
        'version': '1.0.0',
        'features': 30,
        'accuracy': '99.9%',
        'precision': '86%',
        'recall': '82%'
    })


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)