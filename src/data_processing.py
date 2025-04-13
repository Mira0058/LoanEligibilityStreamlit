import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(filepath='data/credit.csv'):
    """
    Load dataset from the given filepath
    """
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    data = df.copy()
    
    # Fill missing values for categorical variables
    data['Gender'].fillna('Male', inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    
    # Fill missing values for numerical variables
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    
    return data

def preprocess_data(df):
    """
    Preprocess the data for model training
    """
    # Make a copy
    data = df.copy()
    
    # Drop Loan_ID column if it exists
    if 'Loan_ID' in data.columns:
        data = data.drop('Loan_ID', axis=1)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Convert columns to appropriate types
    data['Credit_History'] = data['Credit_History'].astype('object')
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype('object')
    
    # Create dummy variables
    data = pd.get_dummies(data, columns=['Gender', 'Married', 'Dependents',
                                         'Education', 'Self_Employed', 'Property_Area'], 
                         dtype=int)
    
    # Replace values in Loan_Approved column
    if 'Loan_Approved' in data.columns:
        data['Loan_Approved'] = data['Loan_Approved'].replace({'Y': 1, 'N': 0})
    
    return data

def split_and_scale_data(df, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets and scale features
    """
    # Separate features and target
    X = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler