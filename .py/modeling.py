import joblib
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from tqdm import tqdm
import hashlib
import copy
import pandas as pd
from datetime import datetime
import json
from sklearn.utils import resample
import pickle

# Load normalized data
X_train_normalized = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_train_normalized.joblib')
X_test_normalized = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_test_normalized.joblib')

# Load standartize data
X_train_standartize = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_train_standarize.joblib')
X_tests_standartize = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_test_standarize.joblib')

# Load target variables
y_train = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/y_train.joblib')
y_test = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/y_test.joblib')

# Load original dataframe
df = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/df.joblib')

# check shape
print('Shape X_train_normalize', X_train_normalized.shape)
print('Shape X_test_normalize', X_test_normalized.shape)
print('Shape X_train_standartize', X_train_standartize.shape)
print('Shape X_train_standartize', X_tests_standartize.shape)
print('Shape y_train', y_train.shape)
print('Shape y_test', y_test.shape)

# Assuming df['price'] is the true target variable
y_true = df['price']

# Baseline prediction
y_pred_baseline = df['price'].mean()

# Calculate Mean Absolute Error (MAE)
mae_baseline = mean_absolute_error(y_true, [y_pred_baseline] * len(y_true))

print('Mean Absolute Error (MAE) for baseline model:', mae_baseline)

import os

# create directory for logs model
log_directory = 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs'
os.makedirs(log_directory, exist_ok=True)

log_path = "D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/training_log.json"

def time_stamp():
    return datetime.now()

def create_log_template():
    """
    Create a template for the training log.

    Parameters :
    ------------
    None

    Returns :
    ---------
        dict: A dictionary template for the training log.
    """
    return {
        "model_name": [],
        "model_uid": [],
        "training_time": [],
        "training_date": [],
        "mae": [],
        "data_configurations": [],
    }

def training_log_updater(logger, log_path):
    """
    Update the training log with the provided logger and save it to a JSON file.

    Parameters:
    -----------
    logger : dict
        Dictionary containing training log information.

    log_path : str
        File path where the training log will be saved in JSON format.

    Returns:
    --------
    pd.DataFrame
        DataFrame representing the updated training log.
    """
    training_log = pd.DataFrame(logger)
    training_log.to_json(log_path, orient="records", lines=True)

    return training_log


# list of model
reg_base = DummyRegressor(strategy = 'mean')
reg_knn = KNeighborsRegressor()
reg_lr = LinearRegression()
reg_dt = DecisionTreeRegressor()

list_of_models = {
    "Normalize_Scaler": [
        {"model_name": reg_base.__class__.__name__, "model_object": reg_base, "model_uid": ""},
        {"model_name": reg_knn.__class__.__name__, "model_object": reg_knn, "model_uid": ""},
        {"model_name": reg_lr.__class__.__name__, "model_object": reg_lr, "model_uid": ""},
        {"model_name": reg_dt.__class__.__name__, "model_object": reg_dt, "model_uid": ""},
    ],
    "Standard_Scaler": [
        {"model_name": reg_base.__class__.__name__, "model_object": reg_base, "model_uid": ""},
        {"model_name": reg_knn.__class__.__name__, "model_object": reg_knn, "model_uid": ""},
        {"model_name": reg_lr.__class__.__name__, "model_object": reg_lr, "model_uid": ""},
        {"model_name": reg_dt.__class__.__name__, "model_object": reg_dt, "model_uid": ""},
    ],
}

def train_eval_model(list_of_model, prefix_model_name, x_train, y_train, data_configuration_name, log_path):
    """
    Train and evaluate a list of machine learning models, update a training log, and save the log to a file.

    Parameters :
    ------------
    list_of_model : list 
        List of dictionaries containing model information.

    prefix_model_name : str 
        Prefix to be added to each model name for identification.

    x_train : array-like 
        Input features for training.

    y_train : array-like 
        Target values for training.

    data_configuration_name : str
        Description of the data configuration used for training.

    log_path : str 
        File path where the training log will be saved.

    Returns :
    ---------
        tuple: A tuple containing the file path of the saved training log and the updated list of models.
    """
    list_of_model = copy.deepcopy(list_of_model)
    logger = create_log_template()

    for model in tqdm(list_of_model):    
        model_name = prefix_model_name + "-" + model["model_name"]

        # Fit the model
        start_time = time_stamp()
        model["model_object"].fit(x_train, y_train)
        finished_time = time_stamp()

        # Convert to seconds
        elapsed_time = (finished_time - start_time).total_seconds()

        # Use predictions on the training data
        y_pred = model["model_object"].predict(x_train)
        
        # Calculate MAE
        mae = mean_absolute_error(y_train, y_pred)

        # Create a unique identifier (one-way encryption)
        plain_id = str(start_time) + str(finished_time)
        chiper_id = hashlib.md5(plain_id.encode()).hexdigest()

        model["model_uid"] = chiper_id

        # Add to the logger
        logger["model_name"].append(model_name)
        logger["model_uid"].append(chiper_id)
        logger["training_time"].append(elapsed_time)
        logger["training_date"].append(str(start_time))
        logger["mae"].append(mae)  # Add a column for MAE
        logger["data_configurations"].append(data_configuration_name)

    # Update the training log and save to a file
    training_log = training_log_updater(logger, log_path)

    return training_log, list_of_model

# Normalize Scaler
training_log_norm, list_of_model_norm = train_eval_model(
                                                    list_of_model=list_of_models['Normalize_Scaler'],
                                                    prefix_model_name='Best_Model',
                                                    x_train=X_train_normalized,
                                                    y_train=y_train,
                                                    data_configuration_name="Normalized Data",
                                                    log_path=log_path
                                                )

list_of_models["Normalize_Scaler"] = copy.deepcopy(list_of_model_norm)

# standart Scaler
training_log_stand, list_of_model_stand = train_eval_model(
                                                    list_of_model=list_of_models['Standard_Scaler'],
                                                    prefix_model_name='Best_Model',
                                                    x_train=X_train_standartize,
                                                    y_train=y_train,
                                                    data_configuration_name="Standartize Data",
                                                    log_path=log_path
                                                )

list_of_models["Standard_Scaler"] = copy.deepcopy(list_of_model_stand)

def join_log_predict(list_of_training_logs, sort_metric):
    """
    Combine multiple training logs into a single DataFrame and sort it based on the specified metric and training time.

    Parameters:
    -----------
    list_of_training_logs : list
        List of DataFrames representing individual training logs.

    sort_metric : str
        The metric based on which the combined log should be sorted.

    Returns:
    --------
    pd.DataFrame
        Combined and sorted training log.
    """
    # Combine multiple training logs into a single DataFrame
    combined_log = pd.concat(list_of_training_logs)

    # Sort the combined log based on the specified metric and training time
    if sort_metric in combined_log.columns:
        combined_log.sort_values(by=[sort_metric, 'training_time'], ascending=[True, True], inplace=True)
    else:
        raise ValueError(f"{sort_metric} not found in the training log columns.")

    # Reset the index
    combined_log.reset_index(drop=True, inplace=True)

    return combined_log

# set list training logs
list_of_training_logs = [training_log_norm, training_log_stand]

# call function
training_res = join_log_predict(list_of_training_logs, sort_metric='mae')
training_res

def get_best_model(training_log_df, list_of_model, sort_metric):
    """
    Get the best-performing model based on a given training log and sort metric.

    Parameters:
    -----------
    training_log_df : pd.DataFrame
        DataFrame representing the training log.

    list_of_model : dict
        Dictionary containing a list of models for different configurations.

    sort_metric : str
        The metric based on which the best model should be determined.

    Returns:
    --------
    model_object : object
        The best-performing model object.
    """
    model_object = None

    # Pick the highest one
    best_model_info = training_log_df.sort_values(by=[sort_metric, 'training_time'], ascending=[True, True]).iloc[0]
    
    # looping data
    for configuration_data in list_of_model:
        for model_data in list_of_model[configuration_data]:
            if model_data["model_uid"] == best_model_info["model_uid"]:
                model_object = model_data["model_object"]
                break
    
    if model_object is None:
        raise RuntimeError("The best model not found in your list of model.")
    
    return model_object

model = get_best_model(training_log_df = training_res, 
                       list_of_model = list_of_models,
                       sort_metric = 'mae')
model

# cross validation
skf = KFold(n_splits=5)

# Define the parameter grid for Decision Tree
params_grid = {
    'criterion': ['poisson', 'friedman_mse', 'squared_error', 'absolute_error'],  
    'splitter': ['best', 'random'],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2', None]  
}

random_search = RandomizedSearchCV(
                                model,
                                param_distributions = params_grid,
                                n_iter = 10,
                                cv = skf,
                                scoring = 'neg_mean_absolute_error',
                                random_state = 42
)

# Add information model to list_of_models for normalize_scaler
list_of_models["Normalize_Scaler"].append({
    "model_name": random_search.__class__.__name__,
    "estimator_name": random_search.estimator.__class__.__name__,
    "model_object": copy.deepcopy(random_search),
    "model_uid": ""
})

# Add information model to list_of_models for standart_scaler 
list_of_models["Standard_Scaler"].append({
    "model_name": random_search.__class__.__name__,
    "estimator_name": random_search.estimator.__class__.__name__,
    "model_object": copy.deepcopy(random_search),
    "model_uid": ""
})

# call function with data normalize scaler
training_log_hypnorm, list_of_model_lr_hypnorm = train_eval_model(
    [list_of_models["Normalize_Scaler"][-1]], # newest
    "hyperparams",
    X_train_standartize,
    y_train,
    "Normalize_hype",
    log_path
)

list_of_models["Normalize_Scaler"][-1] = copy.deepcopy(list_of_model_lr_hypnorm[0])

# call function with data standarize scaler
training_log_hypstand, list_of_model_lr_hyp = train_eval_model(
    [list_of_models["Standard_Scaler"][-1]], # newest
    "hyperparams",
    X_train_standartize,
    y_train,
    "Standart_hype",
    log_path
)

list_of_models["Standard_Scaler"][-1] = copy.deepcopy(list_of_model_lr_hyp[0])

# set list training logs
list_of_training_logs = [training_log_norm, training_log_stand, 
                         training_log_hypstand, training_log_hypnorm]

# call function
training_res = join_log_predict(list_of_training_logs, sort_metric='mae')
training_res

def re_train(X_train, y_train, cross_validation):
    """
    Retrain a Decision Tree model using specified or cross-validated best parameters.

    Parameters :
    ------------
    X_train: pd.DataFrame 
        Training data

    y_train : pd.Series 
        Target values.

    cross_validation : object 
        Cross-validation object to find the best parameters.

    Returns :
    ---------    
    model_retrain : DecisionTreeRegressor
        Retrained Decision Tree model.

    feature_importance : pd.Series
        Feature importance values.

    error : float
        Mean Absolute Error (MAE) of the model.
    """
    # copy data to avoid data leakage
    X_train_copy = X_train.copy()

    # Find the best parameters using cross-validation
    best_rand = cross_validation.fit(X_train_copy, y_train)
    best_params = best_rand.best_params_

    # Initialize the model with the best parameters
    model_retrain = DecisionTreeRegressor(**best_params)

    # Retrain the model using training data
    model_retrain.fit(X_train_copy, y_train)

    # Get feature importance
    feature_importance = pd.Series(model_retrain.feature_importances_, 
                                   index=X_train_copy.columns).\
                                   sort_values(ascending=False)

    # Model prediction
    y_pred = model_retrain.predict(X_train_copy)

    # Calculate error using MAE
    error = mean_absolute_error(y_train, y_pred)

    return model_retrain, feature_importance, error

best_model, feature_importance, error = re_train(X_train = X_train_standartize, 
                                                 y_train = y_train, 
                                                 cross_validation = random_search)
print('Mean Absolute Error from data train', error)
best_model

# show feature importance
feature_importance

# save feature importance
with open("D:/Project_Data/project/Project Pribadi/Deployment_visualization/config/feature_importance.pickle", "wb") as model_file:
    pickle.dump(feature_importance, model_file)

def consistency_model(model, X_train_standartize, y_train):
    """
    Evaluate the consistency of a regression model using bootstrap resampling.

    Parameters :
    ------------
    model : object 
        The regression model to evaluate.
    
    X_train_standartize : pd.DataFrame 
         Standardized features of the training data.
    
    y_train : pd.Series 
         Target variable of the training data.

    Returns :
    --------
    Tuple[bool, str]: A tuple containing a boolean indicating whether the model is consistent
      and a message describing the evaluation result.
    """
    # Number of bootstrap iterations
    num_iterations = 10

    # Store MAE for each iteration
    mae_values = []

    for _ in range(num_iterations):
        # Resample the training data
        X_train_resampled, y_train_resampled = resample(X_train_standartize, y_train)

        # Make predictions on resampled training data
        y_pred_resampled = model.predict(X_train_resampled)

        # Calculate Mean Absolute Error (MAE) on the resampled training data
        mae_resampled = mean_absolute_error(y_train_resampled, y_pred_resampled)
        mae_values.append(mae_resampled)

    # Calculate the standard deviation of MAE values
    mae_std_dev = np.std(mae_values)

    # Define a threshold for consistency (adjust as needed)
    consistency_threshold = 0.1

    # Ensure the standard deviation is below the threshold
    return mae_std_dev < consistency_threshold, f"Model predictions are not consistent. \
           Std Dev: {mae_std_dev}, Threshold: {consistency_threshold}"

# call function
consistency_model(model = best_model, 
                  X_train_standartize = X_train_standartize, 
                  y_train = y_train)

y_pred_test = best_model.predict(X_tests_standartize)

result_mae = mean_absolute_error(y_test, y_pred_test)
print(f'Mean Absolute Error: {result_mae}')

def logging_train_tes(model, X_data, y_data, log_path, data_configuration_name):
    """
    Test a trained model on the test set and update the log.

    Parameters:
    -----------
    model : object
        Trained model object.

    X_data : pd.DataFrame
        Input features.

    y_data : pd.Series
        True labels.

    log_path : str
        File path where the testing log will be saved.

    data_configuration_name : str
        Description of the data configuration used for testing.
    """
    # Initialize an empty list to store the log entries
    log_entries = []

    # Check if the log file already exists
    if os.path.exists(log_path):
        # Read existing log entries
        with open(log_path, 'r') as log_file:
            for line in log_file:
                log_entries.append(json.loads(line.strip()))

    # Model prediction on the data
    y_pred = model.predict(X_data)

    # Calculate Mean Absolute Error (MAE) on the data
    result_mae = mean_absolute_error(y_data, y_pred)

    # Add to the testing log
    log_entry = {
        "model_name": f"{model.__class__.__name__}-Test",
        "model_uid": hashlib.md5(str(time_stamp()).encode()).hexdigest(),
        "training_time": 0,  # You may update this with actual testing time
        "training_date": str(time_stamp()),
        "mae": result_mae,
        "data_configurations": data_configuration_name,
    }
    
    # Add the new log entry to the list
    log_entries.append(log_entry)

    # Update the log and save to a file
    with open(log_path, 'w') as log_file:
        for entry in log_entries:
            log_file.write(json.dumps(entry) + '\n')

    return log_entry

path_logging = "D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/train_test_log.json"

# call function re-train and test data
X_re_train = logging_train_tes(model = best_model,
                               X_data = X_train_standartize,
                               y_data = y_train,
                               log_path = path_logging,
                               data_configuration_name = 'X_train_standartize'
                              )

X_test = logging_train_tes(model = best_model,
                           X_data = X_tests_standartize,
                           y_data = y_test,
                           log_path = path_logging,
                           data_configuration_name = 'X_test_standartize'
                           )

# save model
with open("D:/Project_Data/project/Project Pribadi/Deployment_visualization/best_model.pickle", "wb") as model_file:
    pickle.dump(best_model, model_file)
