import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifact', 'model.pkl')
    

class ModelTrainer:
    
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data...')
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Forest': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbours': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(),
                'AdaBoost': AdaBoostRegressor()
            }
            
            logging.info('Declaring hyper parameters...')
            params = {
                'Random Forest': {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                
                'Decision Forest': {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                },
            
                'Gradient Boosting': {
                    # 'loss':['squared_error','huber','quantile', 'absolute_error'],
                    'learning_rate':[0.1, .01, .001],
                    'subsample':[0.67, 0.7, 0.75, 0.8, 0.9],
                    # 'criterion':['friedman_mse','squarer_error'],
                    # 'max_features':['sqrt','log2'],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                
                'Linear Regression': {
                    # 'fit_intercept':[True,False],
                    # 'normalize':[True,False],
                    # 'copy_X':[True,False]
                },
                
                'K-Neighbours': {
                    'n_neighbors':[3,5,7,9],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['auto','ball_tree','kd_tree','brute']
                },
                
                'XGBoost': {
                    'learning_rate':[0.1, 0.05, 0.01, 0.001],
                    'max_depth':[3, 5, 7, 9],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                
                'CatBoost': {
                    'learning_rate':[0.1, 0.05, 0.01, 0.001],
                    'depth':[3, 5, 7, 9],
                    'iterations':[30, 50, 100, 200]
                },
                
                'AdaBoost': {
                    'learning_rate':[0.1, 0.05, 0.01, 0.001],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                }
            }
            
            model_report:dict = evaluate_model(X_train =X_train, y_train = y_train, X_test=X_test, y_test=y_test, models = models, param=params)
            
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found!!")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj= best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_sc = r2_score(y_test, predicted)
            
            return r2_sc
            
        except Exception as e:
            raise CustomException(e, sys)