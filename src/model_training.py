from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

def create_logistic_regression():
    """
    Create a logistic regression model with default parameters
    """
    return LogisticRegression()

def create_random_forest(n_estimators=100, max_depth=None):
    """
    Create a random forest model with specified parameters
    """
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

def train_model(model, X_train, y_train):
    """
    Train a model on the given data
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, conf_matrix

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation for a model
    """
    cv_scores = cross_val_score(model, X, y, cv=cv)
    return cv_scores.mean(), cv_scores.std()

def save_model(model, filename, directory='models'):
    """
    Save a trained model to disk
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    return filepath

def load_model(filepath):
    """
    Load a trained model from disk
    """
    return joblib.load(filepath)