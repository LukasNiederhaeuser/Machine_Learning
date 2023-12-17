import os
import pandas as pd
import numpy as np

from src.california_housing import read_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def data_preprocessing(data):
    
    # Create a copy of the data
    data_copy = data.copy()
    data_copy = data_copy.drop('median_house_value', axis=1)

    # Define numerical and categorical columns
    categorical_columns = ["ocean_proximity"]
    numerical_columns_standard = ["longitude", "latitude"]
    numerical_columns_minmax = [col for col in data_copy.columns if col not in categorical_columns + numerical_columns_standard]

    # Pipeline for categorical features
    cat_pipeline = Pipeline(steps=[
        ('one-hot-encoding', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Pipeline for standard numerical features
    num_pipeline_standard = Pipeline(steps=[
        ('scale', StandardScaler())
    ])

    # Pipeline for minmax numerical features
    num_pipeline_minmax = Pipeline(steps=[
        ('scale min max', MinMaxScaler())
    ])

    # Create column transformer
    col_trans = ColumnTransformer(transformers=[
        ('cat_pipeline', cat_pipeline, categorical_columns),
        ('num_pipeline_standard', num_pipeline_standard, numerical_columns_standard),
        ('num_pipeline_minmax', num_pipeline_minmax, numerical_columns_minmax)
    ],
        remainder='drop',
        n_jobs=-1)

    return col_trans.fit_transform(data_copy)


def train_linear_regression(X, y):
    
    # Define Model
    lin_reg = LinearRegression()
    lin_reg_pipeline = Pipeline(steps=[
        ('model', lin_reg)
    ])

    # Fit the model
    lin_reg_pipeline.fit(X, y)

    return lin_reg_pipeline


def main():

    # Read train and test data
    df_train = read_data.read_file(folder="california_housing",filename="strat_train_set_adjusted", csv=True)
    df_test = read_data.read_file(folder="california_housing",filename="strat_test_set_adjusted", csv=True)

    # Create train variables
    X_train = data_preprocessing(df_train)
    y_train = df_train[['median_house_value']]

    # Scikit Learn Linear Regression
    lin_reg_model = train_linear_regression(X_train, y_train)

    # Preprocess test data using the same transformations
    X_test = data_preprocessing(df_test)
    y_test = df_test[['median_house_value']]

    # Predict using the trained model
    y_pred = lin_reg_model.predict(X_test)

    return y_test, y_pred, lin_reg_model


if __name__ == '__main__':
    main()