import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from io import StringIO

def main():
    st.title('SVM Model with Grid Search')
    
    # Load Iris dataset
    st.subheader('Load Dataset')
    df = load_iris()
    X = df.data
    y = df.target
    
    # Display dataset info
    st.write("### Dataset Info")
    st.write(f"Number of samples: {X.shape[0]}")
    st.write(f"Number of features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Hyperparameter tuning
    st.subheader('Hyperparameter Tuning')
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': [0.1, 1, 10],
        'degree': [3, 4, 5],
        'coef0': [0, 1, 10]
    }
    
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    st.write("### Best Parameters")
    st.write(grid_search.best_params_)
    
    st.write("### Best Score")
    st.write(grid_search.best_score_)
    
    # Predictions
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    
    st.write("### Accuracy")
    st.write(accuracy_score(y_test, y_pred))
    
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
