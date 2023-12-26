import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Raw folder
FOLDER_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
FILENAME_RAW = "fish.csv"
filename_raw = os.path.join(FOLDER_RAW, FILENAME_RAW)

# Processed folder
FOLDER_PROCESSED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "fish_regression")


def load_data() -> pd.DataFrame:
    
    # Load data and set index
    data = pd.read_csv(filename_raw, delimiter=",")
    
    # Filter out rows where numerical values are 0
    numerical_columns = ["Length1", "Length2", "Length3", "Height", "Width", "Weight"]
    data = data[(data[numerical_columns] != 0).all(axis=1)]
    
    return data


def train_and_test_split(data):

    # Define species --> to perform stratified sample
    species_column = "Species"

    # Split the data into df1 and df2
    df1, df2 = train_test_split(data, test_size=0.2, random_state=42, stratify=data[species_column])

    return df1, df2


def store_data(train, test):
        
    try:
        # Define file paths for train and test sets
        train_set_path = os.path.join(FOLDER_PROCESSED, "strat_train_set.csv")
        test_set_path = os.path.join(FOLDER_PROCESSED, "strat_test_set.csv")

        # Save train and test sets to CSV
        train.to_csv(train_set_path, index=False)
        test.to_csv(test_set_path, index=False)

        print(f"Step 1 of 8: Successfully executed - stratified train and test set saved to: {train_set_path}")

    except Exception as e:
        print(f"Step 1 of 8: An error occurred while saving the data: {e}") 

def main():

    # Load data
    data = load_data()
    # Create train and test split
    train, test = train_and_test_split(data)
    # Save train and test split
    store_data(train, test)

if __name__ == '__main__':
    main()