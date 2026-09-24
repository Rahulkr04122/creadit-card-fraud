"""
Credit Card Fraud Prediction System
Loads trained model and makes predictions on new transactions
"""

import joblib
import pandas as pd
import numpy as np
import os


class FraudPredictor:
    """
    Predicts whether a credit card transaction is fraudulent
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize the predictor by loading model and scaler
        
        Parameters:
        -----------
        model_path : str
            Path to saved model (.pkl file)
        scaler_path : str
            Path to saved scaler (.pkl file)
        """
        print("Loading model and scaler...")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        print("✓ Model and scaler loaded successfully")
    
    def predict_single(self, transaction_data):
        """
        Predict fraud for a single transaction
        
        Parameters:
        -----------
        transaction_data : dict
            Dictionary with transaction features
        
        Returns:
        --------
        dict with prediction results
        """
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Reorder columns to match training data
        feature_order = self.scaler.feature_names_in_
        df = df[feature_order]

        # Set feature names explicitly (fixes warning)
        df.columns = feature_order
    
        
        # Scale the features
        df_scaled = self.scaler.transform(df)
        
        # Get prediction and probability
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]
        
        # Extract fraud probability
        fraud_prob = probability[1]
        
        # Format result
        result = {
            'prediction': int(prediction),
            'fraud': 'Yes' if prediction == 1 else 'No',
            'probability': float(fraud_prob),
            'confidence': float(fraud_prob * 100)
        }
        
        return result
    
    def predict_batch(self, transactions_df):
        """
        Predict fraud for multiple transactions
        
        Parameters:
        -----------
        transactions_df : pd.DataFrame
            DataFrame with multiple transactions
        
        Returns:
        --------
        pd.DataFrame with predictions
        """
        
        # Reorder columns to match training data
        feature_order = self.scaler.feature_names_in_
        transactions_df = transactions_df[feature_order]


        # Set feature names explicitly
        transactions_df.columns = feature_order
        
        # Scale features
        df_scaled = self.scaler.transform(transactions_df)
        
        # Get predictions
        predictions = self.model.predict(df_scaled)
        probabilities = self.model.predict_proba(df_scaled)[:, 1]
        
        # Create result DataFrame
        result_df = transactions_df.copy()
        result_df['Prediction'] = predictions
        result_df['Fraud_Probability'] = probabilities
        result_df['Fraud'] = result_df['Prediction'].apply(lambda x: 'Yes' if x == 1 else 'No')
        
        return result_df


def main():
    """Test the predictor"""
    
    # Initialize predictor
    predictor = FraudPredictor(
        model_path='models/random_forest.pkl',
        scaler_path='models/scaler.pkl'
    )
    
    # Example: Single transaction
    print("\n" + "=" * 80)
    print("EXAMPLE: Single Transaction Prediction")
    print("=" * 80)
    
    transaction = {
        'Time': 1000,
        'V1': -0.5, 'V2': 0.2, 'V3': -1.0, 'V4': 0.5, 'V5': -0.3,
        'V6': 0.1, 'V7': -0.2, 'V8': 0.3, 'V9': -0.4, 'V10': 0.2,
        'V11': -0.1, 'V12': 0.5, 'V13': -0.3, 'V14': 0.2, 'V15': -0.4,
        'V16': 0.1, 'V17': -0.2, 'V18': 0.3, 'V19': -0.5, 'V20': 0.2,
        'V21': -0.3, 'V22': 0.1, 'V23': -0.2, 'V24': 0.4, 'V25': -0.1,
        'V26': 0.2, 'V27': -0.3, 'V28': 0.1,
        'Amount': 50.0
    }
    
    result = predictor.predict_single(transaction)
    
    print(f"\nTransaction Amount: ${transaction['Amount']:.2f}")
    print(f"Prediction: {result['fraud']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']:.2f}%")


if __name__ == "__main__":
    main()