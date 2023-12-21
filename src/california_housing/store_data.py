import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


# Raw folder
FOLDER_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
FILENAME_RAW = "california_housing.csv"
filename_raw = os.path.join(FOLDER_RAW, FILENAME_RAW)

# Processed folder
FOLDER_PROCESSED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "california_housing")


def load_data() -> pd.DataFrame:
    
    """
    Load data from raw-folder and return the dataframe.
    """

    # Load data and set index
    data = pd.read_csv(filename_raw, delimiter=",")
    # Create new column to create startified sample later
    data["income_cat"] = pd.cut(data["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    # Add calculated columns
    data["rooms_per_household"] = data["total_rooms"]/data["households"]
    data["population_per_houshold"] = data["population"]/data["households"]
    
    return data


def train_and_test_split(data):

    """
    Takes data from load_data as input and performs the train and test split on the data.
    """

    # Create split variable and stratified sample
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    # Remove income_cat column
    strat_train_set.drop("income_cat", axis=1, inplace=True)
    strat_test_set.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def store_data(strat_train_set, strat_test_set):
    
    """
    Save stratified train and test sets as CSV files in the processed folder.
    """
    
    try:
        # Define file paths for train and test sets
        train_set_path = os.path.join(FOLDER_PROCESSED, "strat_train_set.csv")
        test_set_path = os.path.join(FOLDER_PROCESSED, "strat_test_set.csv")

        # Save train and test sets to CSV
        strat_train_set.to_csv(train_set_path, index=False)
        strat_test_set.to_csv(test_set_path, index=False)

        print(f"Step 1 of 8: Successfully executed - stratified train and test set saved to: {train_set_path}")

    except Exception as e:
        print(f"Step 1 of 8: An error occurred while saving the data: {e}")


def main():

    # Load data
    data = load_data()
    # Create train and test split
    strat_train_set, strat_test_set = train_and_test_split(data)
    # Save train and test split
    store_data(strat_train_set, strat_test_set)

if __name__ == '__main__':
    main()
