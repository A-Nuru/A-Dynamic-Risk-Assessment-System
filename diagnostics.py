
import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])  

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    #read the deployed model and a test dataset, calculate predictions
    model = load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = load(os.path.join(model_path, "encoder.pkl"))
    
    if dataset_path is None: dataset_path = "testdata.csv"
    df = pd.read_csv(os.path.join(test_data_path, dataset_path))

    df_x, df_y, _ = preprocess_data(df, encoder)

    y_pred = model.predict(df_x)

    return y_pred, df_y

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    
    
    return #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
