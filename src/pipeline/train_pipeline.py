from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys


if __name__ == "__main__":
    try:
        logging.info(">>>> Training pipeline started <<<<")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data transformation completed successfully")

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed successfully with R2 Score: {r2_square}")

        print(f"\n Best model trained successfully!")
        print(f" R2 Score on test data: {r2_square}")
        print(f"Model saved at: artifacts/model.pkl")

        logging.info(">>>> Training pipeline completed successfully <<<<")

    except Exception as e:
        logging.error("Error occurred in training pipeline")
        raise CustomException(e, sys)
