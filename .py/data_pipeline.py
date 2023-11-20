import psycopg2
import pandas as pd
import joblib
from sqlalchemy import create_engine, Integer, Numeric, String, Date, DateTime, Boolean, Float
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Normalizer
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import yaml

def get_data_sql(table_name, database, password, user, host='localhost', port=5432):
    """
    Get column names and data types for a PostgreSQL table.

    Parameters:
    -----------
    table_name : str
        The name of the PostgreSQL table.

    database : str
        The name of the database.

    password : str
        The password for the database user.

    user : str
        The username for the database user.

    host : str
        The host where the PostgreSQL server is running (default is 'localhost').

    port : int 
        The port on which the PostgreSQL server is listening (default is 5432).

    Returns:
    --------
    column_info : dict
        A dictionary containing column names and their data types sqlalchemy.
    """
    try:
        # Establish a connection to the database
        conn = psycopg2.connect(host=host, dbname=database, password=password, user=user, port=port)

        # Create a cursor
        cursor = conn.cursor()

        # Execute a query to get table metadata
        query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
        cursor.execute(query)

        # Fetch data into a dictionary
        column_info = {col[0]: col[1] for col in cursor.fetchall()}

        # Close the cursor and connection
        cursor.close()
        conn.close()

        type_mapping = {
            'integer': Integer,
            'numeric': Numeric,
            'character varying': String,
            'date': Date,
            'timestamp without time zone': DateTime,
            'boolean': Boolean,
            'float': Float
        }

        # change datatype pandas to sqlalchemy
        column_info = {key: type_mapping[value] for key, value in column_info.items()}
        
        return column_info

    except psycopg2.Error as e:
        # Print the error and return None
        print(f"Error connecting to the database: {e}")
        return None
    
def collect_sql(select_query, table_name, database, user, password, host='localhost', port=5432):
    """
    Function for collecting data from the database and converting it to a pdDataFrame
    
    Parameters :
    ------------
    select_query : str
        SQL query for selecting data in the database
    
    Returns :
    -------
    df_duplicate : pd.DataFrame
        DataFrame containing the selected data without duplicates
    """
    try:
        # Get column information
        column_info = get_data_sql(table_name=table_name, database=database, user=user, password=password, host=host, port=port)

        # Establish a connection to the database
        conn = psycopg2.connect(host=host, dbname=database, password=password, user=user, port=port)

        # Create a cursor
        curr = conn.cursor()

        # Execute the SQL query
        curr.execute(select_query)

        # Fetch data into a Pandas DataFrame
        columns = [desc[0] for desc in curr.description]
        data = curr.fetchall()

        # Create the DataFrame
        df = pd.DataFrame(data, columns=columns)
        df_duplicate = df.drop_duplicates(keep='first')

        print('Before Drop Data Duplicate', df.shape)
        print('After Drop Data Duplicate', df_duplicate.shape)

        # Close the cursor and connection
        curr.close()
        conn.close()

        return df_duplicate, column_info

    except psycopg2.Error as e:
        # Print the error and return None
        print(f"Error connecting to the database: {e}")
        return None
    
df, _ = collect_sql(select_query="SELECT * FROM house_cl", database='clean_data', 
                table_name='house_cl', password='nifi', 
                user='nifi', host='localhost', port=5432)
df.head()

# drop column not used
df = df.drop(columns = 'house_id')
print('After drop column', df.shape)

# check data type
df.info()

# change data types ordinal to object
df['bedrooms'] = df['bedrooms'].astype('object')
df['bathrooms'] = df['bathrooms'].astype('object')
df['stories'] = df['stories'].astype('object')
df['parking'] = df['parking'].astype('object')

# sanichek dtypes
df.info()

# describe data
df.describe()

# check distribution

# looping column for check data type
for column in df.columns[:2]:  

    # select column and chart type
    sns.displot(df[column], kde=True)

    # title of distribution
    plt.title(f'Distribution of {column}')
    plt.show()

# check missing values
print('Data missing nulls is', df.isnull().sum())
print('Data missing nan is', df.isna().sum())

# call params data
params_path = "D:/Project_Data/project/Project Pribadi/Deployment_visualization/config/params.yaml"

def load_params(param_dir):
    """
    Function for obtain data parameters

    Parameter :
    -----------
    param_dir : str 
        path to the parameter file
    
    Returns :
    ---------
    params : yaml
        Open yam file
    """
    with open(param_dir, 'r') as file:
        params = yaml.safe_load(file)
        
    return params

# call function
params = load_params(params_path)

def check_data(input_data, params):
    """
    Function for defense data input

    Parameters :
    ------------
    input_data : pd.Dataframe
        Data from source and we want to check each feature data

    params : Yaml
       parameters that we created before to keep data in the range
    
    Returns :
    ---------
        If the condition data is the same as the parameters, no error is raised 
    """
    # Check length of columns
    assert len(input_data.columns) == params["shape_columns_data"][0], "Error: The number of columns is not as expected."

    # check data types
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."
    assert input_data.select_dtypes("int").columns.to_list() == params["int_columns"], "an error occurs in integer column(s)."

    assert input_data.price.between(params["range_price"][0], params["range_price"][1]).sum() == len(input_data), "an error occurs in range price values,"
    assert input_data.area.between(params["range_area"][0], params["range_area"][1]).sum() == len(input_data), "an error occurs in range area values,"

    check_data(df, params)

def split_input_output(data, target_column):
    """
    Function to separate input & output

    Parameters
    ----------
    data: pandas dataframe
        sample data

    target_column: list
      Columns for output

    Returns
    -------
    X: pandas DataFrame
        input

    y: pandas Series
        output
    """
    # Find the output
    y = data[target_column]

    # Find the input
    X = data.drop(target_column, axis=1)

    return X, y

def split_train_test_valid(data, target_column, test_size=0.20, seed=123):
    """
    Function to separate data into train, test, and validation sets

    Parameters
    ----------
    data: pandas DataFrame
        Sample data

    target_column: str
        Column for output

    test_size: float, default=0.20
        Test data proportion

    seed: int, default=123
        Random state

    Returns
    -------
    X_train: pandas DataFrame
        Train input

    X_test: pandas DataFrame
        Test input

    y_train: pandas Series
        Train output

    y_test: pandas Series
        Test output
    """
    X, y = split_input_output(data, target_column)

    # Split the data into test and temp (train + validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=test_size,
                                                      random_state=seed)

    return X_train, X_test, y_train, y_test

def Ohe_encoder(X_train, X_test, y_train, y_test, label_encoder=None):
    """
    One-Hot Encoding for categorical features and label encoding for target variables.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training input data.
    X_test : pd.DataFrame
        Testing input data.
    y_train : pd.Series
        Training target data.
    y_test : pd.Series
        Testing target data.
    label_encoder : LabelEncoder, optional
        A pre-existing label encoder if you have one, default is None.

    Returns:
    --------
    X_train_ohe : pd.DataFrame
        One-hot encoded training input data.
    X_test_ohe : pd.DataFrame
        One-hot encoded testing input data.
    y_train_encoded : pd.Series
        Label encoded training target data.
    y_test_encoded : pd.Series
        Label encoded testing target data.
    """
    # Initialize a label encoder if not provided
    if label_encoder is None:
        label_encoder = LabelEncoder()

    # Concatenate training and testing target data for label encoding
    all_labels = pd.concat([y_train, y_test], axis=0)
    y_all_encoded = label_encoder.fit_transform(all_labels)

    # Split the label encoded data back into training and testing sets
    y_train_encoded = y_all_encoded[:len(y_train)]
    y_test_encoded = y_all_encoded[len(y_train):]

    # One-hot encode the categorical features in both training and testing sets
    X_train_ohe = pd.get_dummies(X_train)
    X_test_ohe = pd.get_dummies(X_test)

    # Find the common columns between the one-hot encoded sets
    common_columns = X_train_ohe.columns.intersection(X_test_ohe.columns)

    # Keep only the common columns in both sets
    X_train_ohe = X_train_ohe[common_columns]
    X_test_ohe = X_test_ohe[common_columns]

    return X_train_ohe, X_test_ohe, y_train_encoded, y_test_encoded

def normalize_data(data, scaler=None):
    """
    This function is used to normalize data using the Normalizer from scikit-learn.

    Parameters:
    ----------
    data : pd.DataFrame
        Input data in the form of a pd.DataFrame (e.g., X_train, X_test, X_valid).
    scaler : Normalizer, optional
        A pre-existing scaler if you have one, default is None.

    Returns:
    -------
    normalized_data : pd.DataFrame
        The normalized data.
    scaler : Normalizer
        The scaler used for normalization.
    """
    # If a scaler is not provided, create a new one
    if scaler is None:
        # Fit scaler during initialization
        scaler = Normalizer()

    # Normalize the data (transform)
    normalized_data = scaler.transform(data)
    normalized_data = pd.DataFrame(normalized_data,
                                   index=data.index,
                                   columns=data.columns)

    return normalized_data, scaler

def standarize_data(data, scaler=None):
    """
    This function is used to normalize data using the Normalizer from scikit-learn.

    Parameters:
    ----------
    data : pd.DataFrame
        Input data in the form of a pd.DataFrame (e.g., X_train, X_test, X_valid).
    scaler : Normalizer, optional
        A pre-existing scaler if you have one, default is None.

    Returns:
    -------
    normalized_data : pd.DataFrame
        The normalized data.
    scaler : Normalizer
        The scaler used for normalization.
    """
    # If a scaler is not provided, create a new one
    if scaler is None:
        # Fit scaler during initialization
        scaler = StandardScaler()
        scaler.fit(data)

    # Normalize the data (transform)
    standarize_data = scaler.transform(data)
    standarize_data = pd.DataFrame(standarize_data,
                                   index=data.index,
                                   columns=data.columns)

    return standarize_data, scaler

def predict(data):
    """
    This function prepares the data for prediction by splitting it into training, testing, and validation sets,
    and applies one-hot encoding to the categorical features in the data.

    Parameters:
    -----------
    data: pd.DataFrame
      Input data bank

    Returns:
    --------
    X_train: Training features.
    y_train: Training labels.
    X_test: Testing features.
    y_test: Testing labels.
    """
    # Split the data into training, testing, and validation sets
    X_train, X_test, y_train, y_test = split_train_test_valid(data=data,
                                                             target_column='price',
                                                             seed=123)

    # Apply one-hot encoding to categorical features in each dataset
    X_train_ohe, X_test_ohe, y_train_encoded, y_test_encoded = Ohe_encoder(X_train, X_test, y_train, y_test)
    
    # scaling data normalize and standard
    X_train_normalize, scaler = normalize_data(data=X_train_ohe)
    X_train_standarize, scaler = standarize_data(data=X_train_ohe)

    # using scaler from data train to avoid data leakage
    X_test_normalize, _ = normalize_data(data=X_test_ohe, scaler=scaler)
    X_test_standarize, _ = standarize_data(data=X_test_ohe, scaler=scaler)

    print('data X_train_normalize', X_train_normalize.shape)
    print('data X_test_normalize', X_test_normalize.shape)
    print('data X_train_standart', X_train_standarize.shape)
    print('data X_test_standart', X_test_standarize.shape)
    print('data y_train', y_train.shape)
    print('data y_test', y_test.shape)


    return X_train_normalize, y_train_encoded, X_test_normalize, y_test_encoded, X_train_standarize, X_test_standarize

X_train_normalize, y_train_encoded, X_test_normalize, y_test_encoded, X_train_standarize, X_test_standarize = predict(data=df)

# Save normalized data
joblib.dump(X_train_normalize, 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_train_normalized.joblib')
joblib.dump(X_test_normalize, 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_test_normalized.joblib')

# Save standart data
joblib.dump(X_train_standarize, 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_train_standarize.joblib')
joblib.dump(X_test_standarize, 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/X_test_standarize.joblib')

# Save target variables
joblib.dump(y_train_encoded, 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/y_train.joblib')
joblib.dump(y_test_encoded, 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/y_test.joblib')

# Save original dataframe
joblib.dump(df, 'D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/df.joblib')
