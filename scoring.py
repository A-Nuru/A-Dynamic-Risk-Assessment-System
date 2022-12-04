from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from joblib import load
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from features import preprocess_data


#Load config.json and gt path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

#Function for model scoring
def score_model(production=False):
    """Function take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    Input: trained model
    Output: latestscore.txt """
    model = load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = load(os.path.join(model_path, "encoder.pkl"))
    
    if production:
        df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    else:
        df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
        print(df)
    df_x, df_y, _ = preprocess_data(df, encoder)
    print(df_x)
    y_pred = model.predict(df_x)
    print(y_pred)
    f1 = metrics.f1_score(df_y, y_pred)
    print(f1)
    with open(os.path.join(model_path, "latestscore.txt"), "w") as score_file:
        score_file.write(str(f1) + "\n")
    
    return f1

if __name__ == "__main__":
    score_model()
