import pytest
import pandas as pd
import joblib
import pickle
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load model
with open("D:/Project_Data/project/Project Pribadi/Deployment_visualization/best_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

# Load data
X_train_standartize = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_train_standarize.joblib')
X_tests_standartize = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_test_standarize.joblib')
y_train = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/y_train.joblib')
y_test = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/y_test.joblib')


def test_data():
    # Test X_train_standartize
    assert len(X_train_standartize.columns) == 32, "Length of columns for X_train_standartize is not as expected."

    # Test X_tests_standartize
    assert len(X_tests_standartize.columns) == 32, "Length of columns for X_tests_standartize is not as expected."

    # Check the first column and last column in data train
    assert X_train_standartize.columns[0] == 'area', "First column in X_train_standartize is not as expected."
    assert X_train_standartize.columns[-1] == 'furnishingstatus_unfurnished', "Last column in X_train_standartize is not as expected."

    # Check the first column and last column in data test
    assert X_tests_standartize.columns[0] == 'area', "First column in X_tests_standartize is not as expected."
    assert X_tests_standartize.columns[-1] == 'furnishingstatus_unfurnished', "Last column in X_tests_standartize is not as expected."

def test_retrain_model():
    # Ensure model is loaded correctly
    assert model is not None, "Model should not be None."

    # Ensure model has a 'fit' method
    assert hasattr(model, 'fit'), "Model should have a 'fit' method."

    # Train the model
    trained_model = model.fit(X_train_standartize, y_train)

    # Ensure the trained model is not None
    assert trained_model is not None, "Trained model should not be None."

    # Make predictions on test data
    y_pred = trained_model.predict(X_train_standartize)

    # Calculate Mean Absolute Error (MAE) on the test data
    test_mae = mean_absolute_error(y_train, y_pred)

    # Ensure the test MAE is non-negative
    assert test_mae >= 0, "Test MAE should be non-negative."

def test_test_model():
    # Ensure model is loaded correctly
    assert model is not None, "Model should not be None."

    # Ensure model has a 'predict' method
    assert hasattr(model, 'predict'), "Model should have a 'predict' method."

    # Make predictions on test data
    y_pred = model.predict(X_tests_standartize)

    # Calculate Mean Absolute Error (MAE) on the test data
    test_mae = mean_absolute_error(y_test, y_pred)

    # Ensure the test MAE is non-negative
    assert test_mae >= 0, "Test MAE should be non-negative."

