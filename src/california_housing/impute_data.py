import os
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_missing_data(data) -> pd.DataFrame:
    
    # Create a copy of the data
    df_copy = data.copy()

    # Extract the columns for imputation
    columns_for_imputation = ["total_rooms", "total_bedrooms"]
    df_columns_for_imputation = df_copy[columns_for_imputation]

    # Drop the columns to be imputed
    df_without_imputed_columns = df_copy.drop(columns=columns_for_imputation)

    # Create an instance of the iterative imputer
    iterative_imputer = IterativeImputer()

    # Impute the selected columns
    df_columns_after_imputation = pd.DataFrame(iterative_imputer.fit_transform(df_columns_for_imputation), columns=columns_for_imputation)

    # Merge the dataframes back and add a new calculated column
    df_after_imputation = pd.concat([df_without_imputed_columns, df_columns_after_imputation], axis=1)
    df_after_imputation["bedrooms_per_room"] = df_after_imputation["total_bedrooms"] / df_after_imputation["total_rooms"]

    return df_after_imputation