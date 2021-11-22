import pandas as pd 
data = pd.read_csv("delaware_anomaly.csv")

# setting up environment
from pycaret.anomaly import *
ano1 = setup(data, ignore_features=['DEPT_NAME', 'MERCHANT', 'TRANS_DT'])

# creating knn model
knn = create_model('knn')

# saving the model and pipeline 
save_model(knn,r'C:\Users\matti\OneDrive\MyProjects\Projects\P6 Anomaly Detection\P_Anomaly_Detection')
# trained model is saved as a pickle file and imported into Power Query for generating anomaly labels (1 or 0)