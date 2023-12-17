import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold

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


def train_random_forest_regressor(X, y):

    # Define Linear Regression model
    forest_reg = RandomForestRegressor()

    # Define the parameter grid for grid search
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]

    # Create 5-fold cross-validation object
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=forest_reg,
                               param_grid=param_grid,
                               scoring='neg_mean_squared_error',
                               cv=cv,
                               return_train_score=True)
    grid_search.fit(X, y)

    # Print the best parameters from grid search
    print("Best Parameters: ", grid_search.best_params_)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    return best_model