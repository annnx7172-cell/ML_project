import os
import sys #Standard Python libraries — os for file/folder operations, sys for system-level info (used in error handling).
from src.exception import CustomException
from src.logger import logging  
## These import custom error handling and logging tools that the tutor built earlier in src/exception.py and src/logger.py. 
##Instead of plain Python errors, this project tracks errors and logs in a structured way.""
import pandas as pd

from  sklearn.model_selection import train_test_split
from dataclasses import dataclass #A Python feature that lets you create simple classes for storing data (used below for config).

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

#from src.components.model_trainer import ModelTrainer
#from src.components.model_trainer import ModelTrainerConfig
#Imports the next steps in the pipeline (data cleaning/transformation, and model training) so this script can call them automatically.

#Config Class
@dataclass #@dataclass auto-generates the __init__ method, so you don't have to write boilerplate code.
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts',"train.csv")
    test_data_path: str= os.path.join('artifacts',"test.csv")
    raw_data_path: str= os.path.join('artifacts',"data.csv")
#This defines where files will be saved — inside an artifacts folder

#Main Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_Data_Ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df= pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
                         
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)  
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_Data_Ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))         



