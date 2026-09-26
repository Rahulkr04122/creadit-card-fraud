"""
Credit Card Fraud Detection - Streamlit App
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from redictor import FraudPredictor

# Page config
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    return FraudPredictor(
        model_path='models/random_forest.pkl',
        scaler_path='models/scaler.pkl'
    )

# Title
st.title("🛡️ Credit Card Fraud Detection")
st.write("Check if a credit card transaction is fraudulent using Machine Learning")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "About", "Project Info"])

with tab1:
    st.header("Make a Prediction")
    
    try:
        # Load predictor
        predictor = load_model()
        
        # Create form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Transaction Amount ($)", value=50.0, step=0.01, min_value=0.0)
            
            with col2:
                time = st.number_input("Time (seconds)", value=1000, step=1, min_value=0)
            
            st.write("### Transaction Features (V1-V28)")
            st.write("*Leave as 0 for default values*")
            
            features = {}
            cols = st.columns(4)
            
            for i in range(1, 29):
                col = cols[(i-1) % 4]
                with col:
                    features[f'V{i}'] = st.number_input(
                        f"V{i}", 
                        value=0.0, 
                        step=0.01,
                        label_visibility="collapsed"
                    )
            
            # Add submit button
            submitted = st.form_submit_button("🔍 Check Transaction", use_container_width=True)

        # Make prediction
        if submitted:
            # Prepare data
            transaction_data = {
                'Time': time,
                'Amount': amount,
                **features
            }
            
            # Show loading message
            with st.spinner("Analyzing transaction..."):
                result = predictor.predict_single(transaction_data)
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if result['fraud'] == 'Yes':
                    st.error("### ⚠️ FRAUD DETECTED")
                    st.error(f"**Confidence: {result['confidence']:.2f}%**")
                else:
                    st.success("### ✓ LEGITIMATE")
                    st.success(f"**Confidence: {result['confidence']:.2f}%**")
            
            with col2:
                st.metric("Fraud Probability", f"{result['probability']:.4f}")
                st.metric("Amount", f"${amount:.2f}")
            
            # Show explanation
            st.info(f"""
            **Prediction Details:**
            - **Result:** {result['fraud']}
            - **Probability:** {result['probability']:.4f}
            - **Confidence:** {result['confidence']:.2f}%
            """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure model files (random_forest.pkl, scaler.pkl) are in the models/ folder")

with tab2:
    st.header("About This Project")
    
    st.write("""
    ### Problem
    Banks process millions of transactions daily. A small percentage (~0.1%) are fraudulent.
    This system automatically detects fraud to protect customers and banks.
    
    ### Dataset
    - **Total Transactions:** 284,807
    - **Fraudulent:** 492 (0.17%)
    - **Legitimate:** 284,315 (99.83%)
    - **Features:** 30 (anonymized for privacy)
    
    ### Solution
    Random Forest classifier trained on historical transaction data.
    Predicts fraud probability for new transactions in real-time.
    
    ### How It Works
    1. Transaction data is scaled (normalized)
    2. Random Forest model analyzes 30 features
    3. Returns fraud probability (0-1)
    4. Threshold 0.5: If probability ≥ 0.5 → Flag as fraud
    """)

with tab3:
    st.header("Project Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("Accuracy", "99.9%")
        st.metric("Precision", "86%")
        st.metric("Recall", "82%")
        st.metric("F1-Score", "84%")
    
    with col2:
        st.subheader("Technologies")
        st.write("""
        - **Python 3.13**
        - **Pandas** - Data processing
        - **Scikit-learn** - ML models
        - **Streamlit** - Web interface
        """)
    
    st.subheader("Models Trained")
    models_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN'],
        'F1-Score': [0.70, 0.72, 0.84, 0.71],
        'Recall': [0.62, 0.70, 0.82, 0.65],
        'Precision': [0.82, 0.75, 0.86, 0.78]
    }
    st.dataframe(models_data, use_container_width=True)
    
    st.subheader("Links")
    col1, col2 = st.columns(2)
    with col1:
        st.write("[GitHub Repository](https://github.com/Rahulkr04122/credit-card-fraud-detection)")
    with col2:
        st.write("[Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")

# Sidebar
st.sidebar.markdown("""
## 🛡️ Fraud Detection System

**Model:** Random Forest Classifier

**Accuracy:** 99.9%  
**Precision:** 86%  
**Recall:** 82%  
**F1-Score:** 84%

---

### Dataset Stats
- Transactions: 284,807
- Fraud cases: 492 (0.17%)
- Legitimate: 284,315 (99.83%)

---

### How to Use
1. Enter transaction amount
2. Enter transaction time
3. (Optional) Enter feature values V1-V28
4. Click "Check Transaction"
5. Get fraud prediction

---

### Features
- Real-time predictions
- Confidence scores
- Mobile-friendly
- Production-ready

---

Made with ❤️ using ML & Streamlit
""")

st.sidebar.divider()
st.sidebar.info("This is a demo. Model trained on Kaggle dataset.")