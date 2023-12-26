from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def data_preprocessing(data):
    
    # Create a copy of the data
    data_copy = data.copy()
    data_copy = data_copy.drop('Weight', axis=1)

    # Define numerical and categorical columns
    categorical_columns = ["Species"]
    numerical_columns = [col for col in data_copy.columns if col not in categorical_columns]

    # Pipeline for categorical features
    cat_pipeline = Pipeline(steps=[
        ('one-hot-encoding', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Pipeline for standard numerical features
    num_pipeline_standard = Pipeline(steps=[
        ('scale', StandardScaler())
    ])

    # Create column transformer
    col_trans = ColumnTransformer(transformers=[
        ('cat_pipeline', cat_pipeline, categorical_columns),
        ('num_pipeline_standard', num_pipeline_standard, numerical_columns)
    ],
        remainder='drop',
        n_jobs=-1)
    
    # Fit and transform the data
    transformed_data = col_trans.fit_transform(data_copy)

    # Get the column names after transformation
    transformed_column_names = col_trans.get_feature_names_out()

    # Return both the transformed data and the column names
    return transformed_data, transformed_column_names