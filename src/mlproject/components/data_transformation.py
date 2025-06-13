import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.utils import save_object
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the preprocessing pipeline for both
        numerical and categorical columns.
        """
        try:
            numerical_columns = [
                "pH Level",
                "Organic Matter (%)",
                "Nitrogen Content (kg/ha)",
                "Phosphorus Content (kg/ha)"
            ]
            categorical_columns = ["District", "Soil Type"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing pipeline to train and test datasets.
        Returns transformed arrays and path to the saved preprocessing object.
        """
        try:
            logging.info("Reading the train and test files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "Potassium Content (kg/ha)"

            input_features_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing on training and test dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Convert to dense arrays if sparse
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Convert target columns to 2D arrays
            target_train = np.array(target_feature_train_df).reshape(-1, 1)
            target_test = np.array(target_feature_test_df).reshape(-1, 1)

            logging.info(f"input_feature_train_arr shape: {input_feature_train_arr.shape}")
            logging.info(f"target_train shape: {target_train.shape}")

            train_arr = np.concatenate([input_feature_train_arr, target_train], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, target_test], axis=1)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
