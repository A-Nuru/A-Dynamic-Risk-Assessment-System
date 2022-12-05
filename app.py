from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import create_prediction_model
import diagnosis 
import predict_exited_from_saved_model
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, outdated_packages_list, execution_time
from scoring import score_model

#Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

#Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """
    Function to make predictions -call the prediction function
    Output: y_pred, list
    """
    dataset_path = request.json.get('dataset_path')
    y_pred, _ = model_predictions(dataset_path)
    return str(y_pred)
    
#Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    """
    Function to score model - check the score of the deployed model by calling the score_model function
    Input: None
    Output: score, string
    """
    score = score_model()
    return str(score)

#Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    """
    Function to check summary statistics - check means, medians, and modes for each column and return a list of all calculated summary statistics
    Input: None
    Output: summary, list of lists
    """
    summary = dataframe_summary()
    return str(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    return #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
