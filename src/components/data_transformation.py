import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

@dataclass
class DataTransformatiomConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformatiomConfig()
        
    def get_data_transformer_object(self):
        '''
            This function will perform data transformation.
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            logging.info("Creating numerical and categorical pipelines...")
            
            ## handling numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            ## handling categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ('inputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical and categorical pipelines created successfully!")
            
            logging.info("Performing pre-processing...")
            ## performing pre-processing
            preprocessor = ColumnTransformer(
                [
                    ('num_pipline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            logging.info("Pre-processing completed successfully!")
            
            return preprocessor
                  
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train and test data read successfully!")

            logging.info("Creating preprocessor object...")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'math_score'
            
            numerical_columns = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing on train and test data...")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]
            
            logging.info('Saving preprocessor object...')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)