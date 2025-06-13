import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import read_sql_data  # <-- FIXED

if __name__ == "__main__":
    try:
        # Step 1: Load full data from MySQL
        df = read_sql_data()
        print(f"Data shape: {df.shape}")  # Debug

        # Step 2: Split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Step 3: Save to CSV
        os.makedirs("artifacts", exist_ok=True)
        train_data_path = "artifacts/train.csv"
        test_data_path = "artifacts/test.csv"
        train_df.to_csv(train_data_path, index=False)
        test_df.to_csv(test_data_path, index=False)

        # Step 4: Transform data
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # Step 5: Train model
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        raise CustomException(e, sys)
