import os

from src.california_housing import read_data
from src.california_housing import store_data
from src.california_housing import impute_data
from src.california_housing import outlier_removal
from src.california_housing import preprocess_train

# Processed folder
FOLDER_PROCESSED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "california_housing")

# --------------------- Training Data --------------------- #
def training_process():

    # Read raw data, perform test-and train split and store data
    store_data.main()
    
    # Read training-data
    df_train = read_data.read_file(folder="california_housing",
                                filename="strat_train_set",
                                csv=True)
    
    # Impute missing values
    try:
        df_train_after_imputation = impute_data.impute_missing_data(df_train)
        print(f"Step 2 of 8: Successfully executed - missing values imputed for training data")
    except Exception as e:
        print(f"Step 2 of 8: An error occurred while imputing values for training data")

    # Remove outliers
    try:
        df_train_after_outlier_removal = outlier_removal.remove_upper_outliers(data=df_train_after_imputation,
                                                                            column="median_house_value")
        # Get contamination
        percentage_outliers = outlier_removal.get_percentage_outliers(df_train_after_outlier_removal, "median_house_value")
        # Compute Isolation Forest to remove outliers
        df_train_after_complete_outlier_removal = outlier_removal.isolation_forest(data=df_train_after_outlier_removal,
                                                                            anomaly_columns=["median_house_value", "median_income"],
                                                                            percentage_outliers=percentage_outliers,
                                                                            save_model=True)
        print(f"Step 4 of 8: Successfully executed - outlier removed for training data")
    except Exception as e:
        print(f"Step 4 of 8: An error occurred while removing outliers for training data")

    # Save processed stratified train set as CSV files in the processed folder.
    try:
        # Define file paths for train sets
        train_set_path = os.path.join(FOLDER_PROCESSED, "strat_train_set_adjusted.csv")
        # Save processed train set to CSV
        df_train_after_complete_outlier_removal.to_csv(train_set_path, index=False)
        print(f"Step 5 of 8: Successfully executed - stratified processed train set saved to: {train_set_path}")
    except Exception as e:
        print(f"Step 5 of 8: An error occurred while saving the adjusted training-data: {e}")


# --------------------- Test Data --------------------- #
def test_process():

    # Read test-data
    df_test = read_data.read_file(folder="california_housing",
                                  filename="strat_test_set",
                                  csv=True)
    
    # Impute missing values
    try:
        df_test_after_imputation = impute_data.impute_missing_data(df_test)
        print(f"Step 6 of 8: Successfully executed - missing values imputed for test data")
    except Exception as e:
        print(f"Step 6 of 8: An error occurred while imputing values for test data")
    
    # Remove outliers
    try:
        df_test_after_outlier_removal = outlier_removal.remove_upper_outliers(data=df_test_after_imputation,
                                                                              column="median_house_value")
        # Compute Isolation Forest to remove outliers
        df_test_after_complete_outlier_removal = outlier_removal.apply_isolation_forest_model_on_test_data(data=df_test_after_outlier_removal,
                                                                                                       anomaly_columns=["median_house_value", "median_income"])
        print(f"Step 7 of 8: Successfully executed - outlier removed for test data")
    except Exception as e:
        print(f"Step 7 of 8: An error occurred while removing outliers for test data")
    
    
    # Save processed stratified test set as CSV files in the processed folder.
    try:
        # Define file paths for test sets
        test_set_path = os.path.join(FOLDER_PROCESSED, "strat_test_set_adjusted.csv")
        # Save processed train set to CSV
        df_test_after_complete_outlier_removal.to_csv(test_set_path, index=False)
        print(f"Step 8 of 8: Successfully executed - stratified processed test set saved to: {test_set_path}")
    except Exception as e:
        print(f"Step 8 of 8: An error occurred while saving the adjusted test-data: {e}")


def main():

    training_process()
    test_process()


if __name__ == '__main__':
    main()


