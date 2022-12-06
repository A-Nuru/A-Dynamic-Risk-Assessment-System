# A-Dynamic-Risk-Assessment-System
## Overview
As a consulting company offering operational and data consulting to clients, there is extreme concern about 
attrition risk: the risk that some of our clients will exit their contracts and decrease the company's revenue. 
Due to the small client management team, client managers are not able to stay in close contact with all of the clients.
Therefore, there is need to estimate the attrition risk of each of the company's clients.
Thus, the task is to create, deploy, and monitor a risk assessment ML model that will estimate the attrition 
risk of each of the company's clients. This will enable the 
client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Regular monitoring of model is important to ensure that it remains accurate and up-to-date. This is because industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that the company can get risk assessments that are as accurate as possible and minimize client attrition.
## Steps
Data ingestion. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.
