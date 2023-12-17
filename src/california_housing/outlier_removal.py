import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest

# Folder for Isolation-Forest Model
FOLDER_IF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "isolation_forest")


def remove_upper_outliers(data, column):

    # Define data
    data_copy = data.copy()
    data_array = np.array(data[[column]])

    # Define / calculate the variables
    upper_quartile = np.percentile(data_array, 75)
    lower_quartile = np.percentile(data_array, 25)
    iqr = upper_quartile - lower_quartile
    upper_whisker = data_array[data_array<=upper_quartile+1.5*iqr].max()

    data = data_copy[data_copy[column] < upper_whisker]

    return data


def get_percentage_outliers(data, column):

    # Define data
    data = np.array(data[[column]])

    # Define / calculate the variables
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)
    iqr = upper_quartile - lower_quartile
    upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
    lower_whisker = data[data>=lower_quartile-1.5*iqr].min()  

    # Calculate amount of outliers
    amount_outliers_upper = [value for value in data if value > upper_whisker]
    amount_outliers_lower = [value for value in data if value < lower_whisker] 
    percentage_outliers = len(amount_outliers_upper + amount_outliers_lower) / len(data)
        
    return round(percentage_outliers,2)


def isolation_forest(data, anomaly_columns, percentage_outliers, save_model=False):

    # Create the model and fit it to the data
    model_IF = IsolationForest(contamination=percentage_outliers, random_state=42)
    model_IF.fit(data[anomaly_columns])

    # Create new column for anomaly score
    data["anomaly_score"] = model_IF.decision_function(data[anomaly_columns])
    data["anomaly"] = model_IF.predict(data[anomaly_columns])

    # Choose a writable directory within your project
    save_model_path = os.path.join(FOLDER_IF, "isolation_forest_model.joblib")

    # Store model if save_model is True
    if save_model:
        try:
            joblib.dump(model_IF, save_model_path)
            print("Isolation-Forest model saved successfully")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

    # Keep observations not considered outlier and drop columns
    data = data[data["anomaly"] == 1]
    data = data.drop(["anomaly", "anomaly_score"], axis=1)

    return data


def apply_isolation_forest_model_on_test_data(data, anomaly_columns):
    
    # Create a copy of the test data
    test_data_copy = data.copy()

    # Load the Isolation-Forest Model
    model = joblib.load(os.path.join(FOLDER_IF, "isolation_forest_model.joblib"))

    # Create new column for anomaly score
    test_data_copy["anomaly_score"] = model.decision_function(test_data_copy[anomaly_columns])
    test_data_copy["anomaly"] = model.predict(test_data_copy[anomaly_columns])

    # Keep observations not considered outliers and drop columns
    test_data_copy = test_data_copy[test_data_copy["anomaly"] == 1]
    test_data_copy = test_data_copy.drop(["anomaly", "anomaly_score"], axis=1)

    return test_data_copy