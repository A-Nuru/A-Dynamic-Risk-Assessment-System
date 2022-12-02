import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

#Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    """
    Function to ingest data
    check, read,aggregate, record and clean the datasets,  and write to an output file"""
    logger.info('starting data ingestion process')
    # check for datasets
    filenames = next(os.walk(input_folder_path), (None, None, []))[2]  # [] if no file

    # compile the datasets together
    data_list = []
    for file in filenames:
        data_list.append(pd.read_csv(os.path.join(input_folder_path, file)))
        print(data_list)
   
   
    data = pd.concat(data_list)

    # remove duplicates
    data = data.drop_duplicates(ignore_index=True)

    # Write to an output file
    data_path = os.path.join(output_folder_path, 'finaldata.csv')
    try:
        data.to_csv(data_path, index=False)
    except FileNotFoundError:
        os.mkdir(output_folder_path)
        data.to_csv(data_path, index=False)
    return data

    
    



if __name__ == '__main__':
    merge_multiple_dataframe()
