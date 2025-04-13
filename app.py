import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import load_data, preprocess_data, split_and_scale_data
from src.model_training import (create_logistic_regression, create_random_forest, 
                               train_model, evaluate_model, cross_validate_model)
from src.visualization import (plot_target_distribution, plot_confusion_matrix, 
                              plot_categorical_analysis, plot_numerical_feature,
                              plot_correlation_heatmap, plot_feature_importance)
from src.utils import get_categorical_columns, get_numerical_columns, check_missing_values

# Page configuration
st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide")

# Title
st.title("Loan Eligibility Prediction")
st.write("This application helps predict loan eligibility based on applicant information.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Data Exploration", "Model Training", "Prediction"])

# Load data
@st.cache_data
def load_and_preprocess_data():
    df = load_data()
    return df

df = load_and_preprocess_data()

# DATA EXPLORATION PAGE
if page == "Data Exploration":
    st.header("Data Exploration")
    
    # Display basic information
    st.subheader("Data Overview")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")
    
    # Display sample data
    if st.checkbox("Show sample data"):
        st.dataframe(df.head())
    
    # Check for missing values
    if st.checkbox("Check missing values"):
        missing = check_missing_values(df)
        st.write(missing)
    
    # Target variable distribution
    st.subheader("Loan Approval Distribution")
    fig = plot_target_distribution(df)
    st.pyplot(fig)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Categorical features
    categorical_cols = get_categorical_columns(df)
    categorical_cols = [col for col in categorical_cols if col != 'Loan_ID' and col != 'Loan_Approved']
    
    if categorical_cols:
        st.write("### Categorical Features")
        selected_cat = st.selectbox("Select categorical feature", categorical_cols)
        fig = plot_categorical_analysis(df, selected_cat)
        st.pyplot(fig)
    
    # Numerical features
    numerical_cols = get_numerical_columns(df)
    numerical_cols = [col for col in numerical_cols if col != 'Loan_Approved']
    
    if numerical_cols:
        st.write("### Numerical Features")
        selected_num = st.selectbox("Select numerical feature", numerical_cols)
        fig = plot_numerical_feature(df, selected_num)
        st.pyplot(fig)
    
    # Correlation heatmap
    if numerical_cols and len(numerical_cols) > 1:
        st.write("### Correlation Heatmap")
        fig = plot_correlation_heatmap(df, numerical_cols)
        st.pyplot(fig)

# MODEL TRAINING PAGE
elif page == "Model Training":
    st.header("Model Training")
    
    # Data preparation
    st.subheader("Data Preparation")
    
    # Model selection
    st.write("### Select Model")
    model_type = st.radio("Choose a model", ["Logistic Regression", "Random Forest"])
    
    # Random Forest parameters if selected
    if model_type == "Random Forest":
        n_estimators = st.slider("Number of trees", 10, 200, 100)
        max_depth = st.slider("Maximum depth", 1, 20, 5)
    
    # Train-test split parameters
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random state", 0, 100, 42)
    
    # Process button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Preprocess data
            processed_df = preprocess_data(df)
            
            # Split and scale data
            X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
                processed_df, test_size, random_state
            )
            
            # Create and train model
            if model_type == "Logistic Regression":
                model = create_logistic_regression()
                model_name = "Logistic Regression"
            else:
                model = create_random_forest(n_estimators, max_depth)
                model_name = "Random Forest"
            
            model = train_model(model, X_train, y_train)
            
            # Evaluate model
            accuracy, conf_matrix = evaluate_model(model, X_test, y_test)
            
            # Cross-validation
            cv_mean, cv_std = cross_validate_model(model, X_train, y_train)
            
            # Display results
            st.success(f"Model trained successfully! Test accuracy: {accuracy:.4f}")
            
            # Model performance
            st.subheader("Model Performance")
            st.write(f"Test accuracy: {accuracy:.4f}")
            st.write(f"Cross-validation accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(conf_matrix)
            st.pyplot(fig)
            
            # Feature importance for Random Forest
            if model_type == "Random Forest":
                st.subheader("Feature Importance")
                feature_names = processed_df.drop('Loan_Approved', axis=1).columns
                fig = plot_feature_importance(model.feature_importances_, feature_names)
                st.pyplot(fig)

# PREDICTION PAGE
elif page == "Prediction":
    st.header("Loan Eligibility Prediction")
    st.write("Enter applicant information to predict loan eligibility.")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
        loan_amount_term = st.selectbox("Loan Amount Term", [360, 180, 120, 60, 300, 480])
        credit_history = st.selectbox("Credit History", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    # Predict button
    if st.button("Predict"):
        # Create a dataframe with the input values
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Prepare a dummy dataframe for preprocessing
        dummy_df = df.copy()
        dummy_df = dummy_df.iloc[0:1]
        for col in input_df.columns:
            dummy_df[col] = input_df[col].values[0]
        
        # Process input data
        processed_input = preprocess_data(dummy_df)
        processed_input = processed_input.drop('Loan_Approved', axis=1)
        
        # For demonstration purposes, use a simple rule-based prediction
        # In a real application, we would load a trained model here
        if credit_history == 1 and applicant_income > 2500:
            prediction = "Approved"
            probability = 0.85
        else:
            prediction = "Rejected"
            probability = 0.75
        
        # Display prediction
        st.subheader("Prediction Result")
        if prediction == "Approved":
            st.success(f"Loan is likely to be {prediction} (Confidence: {probability:.2f})")
        else:
            st.error(f"Loan is likely to be {prediction} (Confidence: {probability:.2f})")
        
        st.info("""
        Note: This is a simplified prediction for demonstration purposes. 
        In a real application, we would use a trained machine learning model for predictions.
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
### About
This application demonstrates loan eligibility prediction based on the notebook example.
""")