

import training
import scoring
import deployment
import diagnostics
import reporting

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

##################Check and read new data
#first, read ingestedfiles.txt
ingested_files =[]
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as report_file:
    for line in report_file:
        ingested_files.append(line.rstrip())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = False
for filename in os.listdir(input_folder_path):
    if input_folder_path + "/" + filename not in ingested_files:
        new_files = True

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







