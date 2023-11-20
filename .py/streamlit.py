import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
import pickle

def load_json(path):
    """
    Function for load data json file

    Parameters :
    ------------
    path : json
        Location for load data with json format and then convert to pd.Dataframe
    """
    # initialization list
    log_entries = []

    # open json data 
    with open(path, 'r') as file_result:
        for line in file_result:
            log_entries.append(json.loads(line.strip()))
    
    # convert to pd.DataFrame
    df = pd.DataFrame(log_entries)

    return df

# load json score mae model
df = load_json(path = "D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/train_test_log.json")

# load data user
df_user = load_json(path = "D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/user_data.json")

# load data df
df = joblib.load('D:/Project_Data/project/Project Pribadi/Deployment_visualization/data_preprocessing/df.joblib')

# log data train
training_data_model = joblib.load("D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/training_log_model.joblib")

# log data cross validation
training_res_cv = joblib.load("D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/training_res_cv.joblib")

# load feature_importance
with open("D:/Project_Data/project/Project Pribadi/Deployment_visualization/config/feature_importance.pickle", "rb") as model_file:
    feature_importance = pickle.load(model_file).head(5)

# font_size
font_size = 20

# set pages of dashboard
st.set_page_config(
    page_title="Monitoring Model House Pricing",
    page_icon="✅",
    layout="wide",
)

# setting font with CSS 
st.markdown(f"""<style>
            input[type="number"] {{
            font-size: {font_size}px;
            }}
            </style>""", unsafe_allow_html=True)

def prediction():
    """
    Function for input data by user and then create prediction from FastAPI

    Parameters :
    ------------
    None

    Result :
    --------
    Success : json.format 
        Data input from the user will be predicted in the fastApi, and if successful, 
    connect streamlit and fastApi, then will show the result prediction.
    
    Error : print.Text
        If streamlit does not connect with FastApi, then show the text error.
    """
    st.title('Home Pricing Prediction 🏠')
   
    # Data input numeric
    area = st.number_input('Area', min_value=0.0, max_value=10000.0, value=0.0)

    # Define categorical features and their options
    categorical_features = [
        ("Bedrooms", ["1", "2", "3", "4", "5"]),
        ("Bathrooms", ["1", "2", "3"]),
        ("Stories", ["1", "2", "3", "4"]),
        ("Mainroad", ["Yes", "No"]),
        ("Guest Rooms", ["Yes", "No"]),
        ("Basement", ["Yes", "No"]),
        ("Hot water heating", ["Yes", "No"]),
        ("Airconditioning", ["Yes", "No"]),
        ("Parking", ["2", "3", "0", "1"]),
        ("Prefarea", ["Yes", "No"]),
        ("Furnishingstatus", ["furnished", "semi_furnished", "unfurnished"]),
    ]

    # Dictionary to store the data
    data = {}

    for feature, options in categorical_features:
        # Create box in Streamlit
        selected_option = st.selectbox(f"Select {feature}", options)

        # One-hot encode the selected option
        encoded_options = [1 if opt == selected_option else 0 for opt in options]

        # Result saved to data dictionary
        feature_name = f"{feature}_{selected_option}"
        data[feature_name] = 1
        for opt, value in zip(options, encoded_options):
            if opt != selected_option:
                other_feature_name = f"{feature}_{opt}"
                data[other_feature_name] = 0

    # Add numerical features to the data dictionary
    data["area"] = area

    if st.button("Submit"):

        # Make sure all required fields are present in the data dictionary
        required_fields = [
            "GuestRoom_Yes", "GuestRooms_No",
            "HotWaterHeating_Yes", "HotWaterHeating_No",
            "furnishingstatus_furnished", "furnishingstatus_semi_furnished", 
            "furnishingstatus_unfurnished"
        ]
        for field in required_fields:
            if field not in data:
                data[field] = 0  # Set to 0 if not selected
    # input name
    st.sidebar.text('*You must fills your biodata!')
    
    # Access global variables
    global name, email, region, work, salary, result
    # input name user
    name = st.sidebar.text_input('*Input Your Name')

    # Email validation
    email = st.sidebar.text_input('*Input Your Email')
    if "@" not in email:
        st.sidebar.warning("Please enter a valid email address.")

    # input region user
    region = st.sidebar.text_input('*Input Your Region')

    # input work user
    work = st.sidebar.text_input('*Input Your Work')

    # input salary user
    salary = st.sidebar.number_input('*Input Your Salary $')
    
    # Make POST request to FastAPI server
    try:
        # Check biodata
        if not (name and email and region and salary and work):
            st.warning("Please fill in all biodata fields before submitting.")
        else:
            response = requests.post("http://localhost:8000/prediction", json=data)
            response.raise_for_status()
            result = response.json()

            # get values json result
            prediction_value = result.get("prediction", "Error: No prediction available")

            # show values prediction
            st.markdown(f"**Your Home Price $:** <span style='color: green;'>{prediction_value}</span>",
                        unsafe_allow_html=True)

            # save log input user
            log_path_user = "D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/user_data.json"
            
            # initialization list
            all_user_data = []

            # open file 
            try:
                with open(log_path_user, "r") as json_file:
                    for line in json_file:
                        all_user_data.append(json.loads(line.strip()))
            except FileNotFoundError:
                pass

            # Add new user data to the list
            user_data = {
                "name": name,
                "email": email,
                "region": region,
                "work": work,
                "salary": salary,
                "Prediction" : prediction_value
            }

            all_user_data.append(user_data)

            # Save the updated list to the file
            with open(log_path_user, "w") as json_file:
                for entry in all_user_data:
                    json_file.write(json.dumps(entry) + '\n')

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to make prediction. Error: {e}")

def dashboard_model():
    """
    Function for monitoring performance model

    Parameters :
    ------------
    None

    Returns :
    ---------
    visualization : lineplot and table
    """
    # title page
    st.title('Model Monitoring Performance  📈')

    # Save data to list
    log_entries = []

    with open("D:/Project_Data/project/Project Pribadi/Deployment_visualization/logs/train_test_log.json", 'r') as file_result:
        for line in file_result:
            log_entries.append(json.loads(line.strip()))

    # Convert to DataFrame
    df = pd.DataFrame(log_entries)

    # Split data based on data_configurations columns
    df_train = df[df.data_configurations == 'X_train_standartize']
    df_test = df[df.data_configurations == 'X_test_standartize']

    # Sort DataFrame by training_date
    df_train = df_train.sort_values(by='training_date', ascending=False)
    df_test = df_test.sort_values(by='training_date', ascending=False)

    # Take the five latest entries
    df_train_latest = df_train.head(5)
    df_test_latest = df_test.head(5)

    # Change DataFrame to Series
    train_mae_latest = df_train_latest.mae.values
    test_mae_latest = df_test_latest.mae.values
    time_latest = df_train_latest.training_date.values

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))

    # Plot lines
    ax.plot(time_latest, test_mae_latest, color="green", label='Test_Score')
    ax.plot(time_latest, train_mae_latest, color="red", label='Train_Score')

    # Annotate Test plot with values
    for x, y in zip(time_latest, test_mae_latest):
        ax.annotate(f'{y:.2f}', (x, y), 
                    textcoords="offset points", xytext=(0, 5), 
                    ha='center', fontsize=10, color='black')

    # Annotate Train plot with values
    for x, y in zip(time_latest, train_mae_latest):
        ax.annotate(f'{y:.2f}', (x, y), 
                    textcoords="offset points", xytext=(0, 5), 
                    ha='center', fontsize=10, color='black')

    # Show legend
    ax.legend()

    # Show the plot
    plt.show()

    # show ylabel and hide values y-axis
    plt.ylabel('mean_absolute_error (MAE)')
    ax.set_yticklabels([])
    
    # remove inner square
    sns.despine()

    # Adjust layout
    st.subheader('Comparison of the MAE Score Train and Test Model DecisionTreeRegressor')
    
    # show plot monitoring MAE
    st.pyplot(fig)

    fig1, ax1 = plt.subplots(figsize=(20, 8))

    # Input data on barplot
    ax1 = sns.barplot(
        x=feature_importance.index,
        y=feature_importance.values,
        palette=sns.color_palette("Blues_r"),
    )

    # create annotation values
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    # remove box inner plot
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    # Remove values y and label X,Y
    ax1.set(xlabel=None, ylabel=None)
    ax1.tick_params(left=False)
    ax1.set_yticklabels([])

    # title chart
    st.title("Monitoring Best Feature Model")

    # show plot feature importance
    st.pyplot(fig1)

    # show table result from cross validation
    st.subheader('Log Result After Cross-Validation')
    st.write("After performing cross-validation using RandomSearchCV, the table below \
             shows the Mean Absolute Error (MAE) values for each fold. A smaller MAE \
             indicates better model performance.")
    
    # show table cross validation score
    st.table(training_res_cv)

    # show table log all model
    st.subheader('Log Result After Training Models')
    st.table(training_data_model)

    # note
    st.sidebar.write("Note !")
    st.sidebar.write("The Model is not consisten in the new Data")

def data_user():
    """
    Function for monitoring data user and save to json format
    """
    st.title("Welcome Result Prediction Data from User 🔍")

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))

    # histogram
    sns.histplot(data = df_user , x = 'salary', kde=True, palette='deep')
    plt.title('Monitoring Salary Client', fontsize = 20 )

    # remove inner square
    sns.despine()

    st.pyplot(fig)

    st.write('In the below, we can see biodata from users who filled out or made predictions, with \
             this data, we can create analysis or know user characteristics with know mean of predicted result\
             We can offer more targeted products based on surveys, or we can create new machine learning\
             classification for knowing with house pricing like this, whether the person will buy')
    
    # find tendency
    mean = df_user.Prediction.mean()
    median = df_user.Prediction.median()

    # Calculate the percentage difference
    percentage_difference = abs((mean - median) / mean) * 100

    # select tendency for mean result
    if percentage_difference <= 5:
        st.markdown(f'<p style="color: black; font-size: 18px; background-color: white;">Mean of prediction value is : {median} --> Tendency Median</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color: black; font-size: 18px;background-color: white;">Mean of prediction value is : {mean} --> Tendency Mean</p>', unsafe_allow_html=True)

    st.table(df_user.head(50))
    st.write('Sum of data user is', df_user.shape[0])


def monitoring_EDA(data, col_cat, col_values, column, plot_color='lightblue'):
        """
        Function for visualization EDA, and then we can know if we have a bad model

        Parameters :
        ------------
        data : pd.Dataframe
            Data without scaling data
        
        col_cat : str or object
            Columns that have category data
        
        column : int
            Column for create distribution data
        
        plot_colors : default

        Returns :
        ---------
            Return visualization barplot, heatmap, and histogram 
        """
        # Code for create barplot visualization
        # grouping and sort data
        df = data.groupby([col_cat])[col_values].sum().reset_index()\
                          .sort_values(by=col_values, ascending=False)

        # setting size visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        # Input data on barplot
        ax = sns.barplot(
                        x=df[col_cat],
                        y=df[col_values],
                        palette=sns.color_palette("Blues_r"),
                        )

        # create annotation values
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

        # Create title visualization
        plt.title(f'Analysis {col_cat} VS {col_values}', fontsize=20)

        # remove box inner plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Remove values y and label X,Y
        ax.set(xlabel=None, ylabel=None)
        ax.tick_params(left=False)
        ax.set_yticklabels([])

        # Create for histogram visualization
        # Create Matplotlib figure and axis
        fig2, ax = plt.subplots(figsize=(12, 8))

        # Create histogram using seaborn
        sns.histplot(data[column], kde=True, ax=ax)

        # Title of distribution
        plt.title(f'Distribution of {column}', fontsize = 20)

        # Code for create heatmap visualization
        # we using 'Spearman' because not sensitive with outlier
        corr = data.corr(method='spearman')
        fig3, ax = plt.subplots(figsize=(12, 8))

        # Buat visualisasi heatmap
        sns.heatmap(corr, cmap='coolwarm', annot=True, vmin=-1, vmax=1, linewidths=0.5)

        # Atur judul
        plt.title('Heatmap Correlation')

        # shape current data
        st.sidebar.write('Data that was used', data.shape)

        # open file visualization
        url = 'D://Project_Data/project/Project Pribadi/Deployment_visualization/Last Update Visualization EDA/'

        if st.sidebar.button('Last Update Visualization'):
            os.system(f'explorer "{url}"')
        
        return fig, fig2, fig3

def main():
    """
    Function for selecting the page: Prediction, Monitoring Model, Monitoring EDA, Data User
    """
    st.sidebar.title("Navigator")
    page_options = ["Prediction", "Monitoring Model", "Monitoring EDA", "Data User"]
    selected_page = st.sidebar.selectbox("Select Page", page_options)

    # Set page query parameter based on selection
    if selected_page == "Prediction":
        page = 1
    elif selected_page == "Monitoring Model":
        page = 2
    elif selected_page == "Monitoring EDA":
        page = 3
    elif selected_page == "Data User":
        page = 4

    # Display content based on selected page
    if page == 1:
        prediction()
    elif page == 2:
        dashboard_model()
    elif page == 3:

        # Title dashboard
        st.title("Monitoring Characteristic Data 📊")
        col1, col2 = st.columns(2)
        with col1:
            # Pass appropriate arguments to monitoring_EDA function
            col_cat = st.selectbox('Select Categorical Column:', df.select_dtypes(include='object').columns, key='col_cat')
        with col2:
            col_values = st.selectbox('Select Numerical Column:', df.select_dtypes(include=['int64', 'float64']).columns, key='col_values')

        column = col_values  # Assuming you want to use the same column for 'column' argument

        # call function and get the Matplotlib figures
        barplot, histogram, heatmap = monitoring_EDA(df, col_cat, col_values, column)

        # Display the figures using st.pyplot()
        plt1, plt2 = st.columns(2)
        
        # call barplot
        with plt1:
            st.pyplot(barplot)
        
        # call histogram
        with plt2:
            st.pyplot(histogram)
        
        # call heatmap
        st.pyplot(heatmap)

    elif page == 4:
        # Password protection for Page 4
        password = st.sidebar.text_input("Enter Password", type="password")
        correct_password = "datauser"

        # condition if correct password
        if password == correct_password:
            st.sidebar.success("Password is correct. You can access Page Data User.")
            data_user()
        
        # condition if incorrect password
        elif password != "":
            st.sidebar.warning("Incorrect password. Please try again.")

# Run the app
if __name__ == "__main__":
    main()


