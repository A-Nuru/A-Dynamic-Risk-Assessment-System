
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

def model_predictions():
    """
    Function to get model predictions - read the deployed model and a test dataset, calculate predictions
    Input: deployed model, test dataset, 
    output: predictions
    """
    model = load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = load(os.path.join(model_path, "encoder.pkl"))
    
    if dataset_path is None: dataset_path = "testdata.csv"
    df = pd.read_csv(os.path.join(test_data_path, dataset_path))

    df_x, df_y, _ = preprocess_data(df, encoder)

    y_pred = model.predict(df_x)

    return y_pred, df_y

def dataframe_summary():
    """Function to calculate the summary statistics
    Otput: list containing all summary statistics
    """
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
        ]
    
    result = []
    for column in numeric_columns:
        result.append([column, "mean", df[column].mean()])
        result.append([column, "median", df[column].median()])
        result.append([column, "standard deviation", df[column].std()])
    return result

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





    
