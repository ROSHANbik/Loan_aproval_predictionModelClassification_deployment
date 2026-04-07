# import the manipulation labraries
import pandas as pd 
import numpy as np 
# step1: data_ingestion 
def data_ingestion():
    df = pd.read_csv('D:\Loan_aproval_predictionModelClassification\data\loan_approval_dataset.csv')
    df.columns
    return df