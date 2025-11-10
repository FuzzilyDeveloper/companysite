import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="ML Model Predictor", layout="wide")

# Title
st.title("ML Model Predictor")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())
    
    # Feature selection
    st.subheader("Select Features")
    features = st.multiselect("Choose features for prediction", data.columns.tolist())
    target = st.selectbox("Choose target variable", data.columns.tolist())
    
    if st.button("Train Model"):
        # Basic preprocessing
        le = LabelEncoder()
        X = data[features].copy()
        y = data[target].copy()
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']):
            X[col] = le.fit_transform(X[col])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Show feature importance
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        st.bar_chart(importance_df.set_index('Feature'))
        
        # Allow new prediction
        st.subheader("Make New Prediction")
        new_data = {}
        for feature in features:
            if data[feature].dtype == 'object':
                new_data[feature] = st.selectbox(f"Select {feature}", data[feature].unique())
            else:
                new_data[feature] = st.number_input(f"Enter {feature}", value=float(data[feature].mean()))
        
        if st.button("Predict"):
            # Prepare input
            input_df = pd.DataFrame([new_data])
            for col in input_df.select_dtypes(include=['object']):
                input_df[col] = le.fit_transform(input_df[col])
            
            # Make prediction
            prediction = model.predict(input_df)
            st.success(f"Prediction: {prediction[0]}")
else:
    st.info("Please upload a CSV file to begin")