import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def run_train_pipeline(self):
        try:
            # Initialize DataTransformation
            data_transformation = DataTransformation()

            # Load train and test data
            train_data = data_transformation.load_data(self.train_data_path)
            test_data = data_transformation.load_data(self.test_data_path)

            # Perform data transformation
            train_features, train_labels = data_transformation.transform_data(train_data)
            test_features, test_labels = data_transformation.transform_data(test_data)

            # Initialize ModelTrainer
            model_trainer = ModelTrainer()

            # Train the model
            trained_model = model_trainer.train_model(train_features, train_labels)

            # Evaluate the model
            evaluation_result = model_trainer.evaluate_model(trained_model, test_features, test_labels)

            logging.info("Training pipeline completed successfully.")

            return trained_model, evaluation_result

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Assuming train and test data paths are obtained from DataIngestion
        train_data_path = "artifacts/train.csv"
        test_data_path = "artifacts/test.csv"

        # Initialize TrainPipeline
        train_pipeline = TrainPipeline(train_data_path, test_data_path)

        # Run the training pipeline
        trained_model, evaluation_result = train_pipeline.run_train_pipeline()

        # Print or use the trained model and evaluation results
        print("Trained model:", trained_model)
        print("Evaluation result:", evaluation_result)

    except CustomException as ce:
        print("Custom Exception occurred:", ce)
    except Exception as e:
        print("Error occurred:", e)
