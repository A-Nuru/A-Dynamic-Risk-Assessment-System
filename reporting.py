import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



#Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path']) 

def score_model():
    """
    Function for reporting - calculate a confusion matrix using the test data and the deployed model, and write the confusion matrix to the workspace"""
    y_pred, df_y = model_predictions(None)
    df_cm = metrics.confusion_matrix(df_y, y_pred)
    

if __name__ == '__main__':
    score_model()
