import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd

def plot_target_distribution(df):
    """
    Plot the distribution of the target variable (Loan_Approved)
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Loan_Approved', data=df)
    plt.title('Loan Approval Distribution')
    plt.xlabel('Loan Approved')
    plt.ylabel('Count')
    
    # Add count labels
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 5, 
                '{:1.0f}'.format(height), ha="center")
    
    return plt

def plot_confusion_matrix(conf_matrix):
    """
    Plot a confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def plot_categorical_analysis(df, column):
    """
    Plot the relationship between a categorical variable and loan approval
    """
    plt.figure(figsize=(10, 6))
    
    # Create crosstab and calculate proportions
    crosstab = pd.crosstab(df[column], df['Loan_Approved'])
    crosstab_perc = crosstab.div(crosstab.sum(axis=1), axis=0)
    
    # Plot
    crosstab_perc.plot(kind='bar')
    plt.title(f'Loan Approval Rate by {column}')
    plt.xlabel(column)
    plt.ylabel('Proportion')
    plt.legend(['Not Approved', 'Approved'])
    plt.xticks(rotation=45)
    
    return plt

def plot_numerical_feature(df, column):
    """
    Plot the distribution of a numerical feature by loan approval status
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue='Loan_Approved', multiple='stack', bins=20)
    plt.title(f'Distribution of {column} by Loan Approval Status')
    plt.xlabel(column)
    plt.ylabel('Count')
    return plt

def plot_correlation_heatmap(df, numerical_columns):
    """
    Plot correlation heatmap for numerical variables
    """
    plt.figure(figsize=(10, 8))
    corr = df[numerical_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    return plt

def plot_feature_importance(importance, features):
    """
    Plot feature importance for a random forest model
    """
    indices = np.argsort(importance)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance - Random Forest')
    return plt